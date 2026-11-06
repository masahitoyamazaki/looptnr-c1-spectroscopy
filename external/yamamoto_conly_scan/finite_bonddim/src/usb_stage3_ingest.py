"""Stream selected ZIP CSV members into a small, derived spectral table.

No extract/extractall, no tensor deserialization, no archived code execution.
"""
import csv,io,json,hashlib,ctypes,os,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data'/'stage3_usb'
OUT.mkdir(parents=True,exist_ok=True)
from zipfile import ZipFile
INV=json.loads((ROOT/'data/usb_zip_inventory.json').read_text(encoding='utf-8'))

def memory():
    class Mem(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(s,ctypes.c_ulonglong) for s in ('total','avail','page_total','page_avail','virtual_total','virtual_avail','extended')]
    class Proc(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(s,ctypes.c_size_t) for s in ('peak','working','qpp','pp','qnp','np','page','peak_page','private')]
    m=Mem();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)): raise OSError('Memory query failed')
    p=Proc();p.cb=ctypes.sizeof(p)
    ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(p),p.cb): raise OSError('Process memory query failed')
    return dict(available_bytes=m.avail,total_bytes=m.total,working_bytes=p.working,peak_working_bytes=p.peak,private_bytes=p.private)

def guard(start=False):
    m=memory()
    # CSV-only budget; tensor deserialization is not part of this program.
    if m['available_bytes']<(1024 if start else 512)*1024**2 or m['working_bytes']>128*1024**2: raise MemoryError(m)
    return m

class HashedReader(io.RawIOBase):
    def __init__(self,stream,limit): self.stream=stream;self.limit=limit;self.count=0;self.sha=hashlib.sha256()
    def readable(self): return True
    def readinto(self,b):
        data=self.stream.read(min(len(b),65536)); self.count+=len(data)
        if self.count>self.limit: raise ValueError('Member byte limit exceeded')
        self.sha.update(data);b[:len(data)]=data;return len(data)

def writecsv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    before=guard(start=True);sources=[];spectra=[];observables=[];settings=[]
    campaigns=[(28,'CU1Irrep','_provenance/chi28_flow_20261020/'),
               (28,'U1Irrep','_provenance/chi28_u1_flow_20261020/'),
               (40,None,'_provenance/chi40_dual_flow_20261020/results/')]
    requests=[(chi,sym,prefix+kind+'.csv',kind) for chi,sym,prefix in campaigns for kind in ('spectra','observables')]
    requests += [(0,None,name,'settings') for name in ('_provenance/chi28_flow_20261020/settings.txt','_provenance/chi28_u1_flow_20261020/settings.txt','_provenance/chi40_dual_flow_20261020/settings.toml')]
    for chi,sym,name,kind in requests:
        matches=[(a,m) for a in INV['archives'] for m in a['members'] if m['name']==name]
        if len(matches)!=1: raise ValueError((name,len(matches)))
        archive,entry=matches[0]
        if entry['uncompressed_bytes']>64*1024**2 or archive['central_directory_bytes']>16*1024**2: raise ValueError('Size budget exceeded')
        n=0; kept=0
        with ZipFile(archive['path']) as z, z.open(name) as raw:
            hashed=HashedReader(raw,entry['uncompressed_bytes'])
            with io.TextIOWrapper(io.BufferedReader(hashed,buffer_size=65536),encoding='utf-8-sig',newline='') as text:
                if kind=='settings':
                    settings.append(dict(name=name,text=text.read(65536)))
                else:
                    for r in csv.DictReader(text):
                        n+=1
                        if n%1000==0: guard()
                        common=dict(chi=chi,symmetry=sym or r['symmetry'],vertex_c=float(r['vertex_c']),step=int(r['rg_step']))
                        if kind=='spectra':
                            q=float(r['charge']);rank=int(r['rank'])
                            if not ((q==0 and rank<=3) or (0<abs(q)<=4 and rank==1)): continue
                            row=dict(**common,charge=q,parity=int(r['parity']),rank=rank,
                                x=float(r['x']),lambda_re=float(r['lambda_real']),lambda_im=float(r['lambda_imag']),
                                residual=float(r['residual']),phase=float(r['lambda_ratio_phase']),
                                selected=r['selected'].lower()=='true',converged=int(r['converged']),status=r['status'])
                            spectra.append(row)
                        else:
                            fields=('central_charge','central_imag','K_exact','tau_tm_real','tau_tm_imag','warning_count')
                            row=dict(**common,**{k:float(r[k]) for k in fields},status=r['status'])
                            observables.append(row)
                        kept+=1
            if hashed.count!=entry['uncompressed_bytes']: raise ValueError('Incomplete read')
        sources.append(dict(member=name,archive=Path(archive['path']).name,bytes_read=hashed.count,sha256=hashed.sha.hexdigest(),crc32=entry['crc32'],rows=n,selected_rows=kept))
        print(name,n,kept,flush=True)
    writecsv(OUT/'selected_spectra.csv',spectra);writecsv(OUT/'selected_observables.csv',observables)
    after=guard()
    (OUT/'source_manifest.json').write_text(json.dumps(dict(read_mode='streaming; no disk extraction; no jls read',memory_before=before,memory_after=after,sources=sources,settings=settings),indent=2),encoding='utf-8')
    print(json.dumps(dict(memory=after,spectral_rows=len(spectra),observable_rows=len(observables)),indent=2))

if __name__=='__main__':main()
