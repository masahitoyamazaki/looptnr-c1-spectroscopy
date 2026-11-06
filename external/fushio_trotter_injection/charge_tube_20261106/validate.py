"""Check provenance, coverage, solver evidence, formulas, links and local copies."""
import hashlib,json,re,tomllib,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[5]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
meta=json.loads((H/'control/inputs.json').read_text())
source_hashes=json.loads((H/'control/source_hashes.json').read_text())
for p,h in source_hashes.items():assert sha(Path(p))==h,p
for source in meta['sources']:
    assert sha(ROOT/source['path'])==source['sha256']
    for c in source['checkpoints']:assert sha(ROOT/c['path'])==c['sha256']
rows=json.loads((H/'analysis.json').read_text())['rows'];wide=[r for r in rows if r['tube']=='141']
assert len(wide)==27
assert {(r['h'],r['step']) for r in wide}=={(h,n) for h in (0.,.01,.02) for n in range(0,17,2)}
for r in wide:
    old=next(v for v in rows if v['tube']=='121' and v['h']==r['h'] and v['step']==r['step'])
    assert r['checkpoint_sha256']==old['checkpoint_sha256']
    assert r['numeric_pass']
    assert abs(r['x1_minus_cft']-(r['x'][0]-1/(4*r['K'])))<1e-14
    assert r['x1_minus_cft']>0  # The plotted signed difference supports a log axis.
    d=tomllib.loads((ROOT/r['source']).read_text());assert d['shape']==[1,4,1]
    assert d['measurement_source_sha256'] in source_hashes.values()
    prior=tomllib.loads((ROOT/old['source']).read_text())
    prior_m=next(m for m in prior['measurements'] if m['step']==r['step'])
    assert abs(d['tau_real']-prior_m['tau_real']/2)<1e-13
    assert abs(d['tau_imag']-prior_m['tau_imag']/2)<1e-13
    assert d['physical_ring_L']==8*2**(r['step']//2)
    assert d['Q_time_extent']==4*2**(r['step']//2)
    for s in d['sectors']:
        assert len(s['trials'])==2 and min(t['converged'] for t in s['trials'])>=1
        assert max(t['residual'] for t in s['trials'])<1e-8
    if r['h']:
        b=next(v for v in wide if v['h']==0 and v['step']==r['step'])
        aa=r['coefficients']['alpha'];AX=r['coefficients']['AX']
        for i,k in enumerate(['m1','m2','m3']):assert abs(r['dG'][k]-(r['x'][i]-b['x'][i])/aa[i])<1e-14
        dx=[x-y for x,y in zip(r['x'],b['x'])]
        assert abs(r['dG']['X']-(15*dx[0]-6*dx[1]+dx[2])/AX)<1e-14
geometry=tomllib.loads((H/'control/geometry_validation.toml').read_text())
assert len(geometry['checks'])==6
geoerr=max(max(c['charge_ratio_errors']) for c in geometry['checks']);assert geoerr<1e-11
sliceerr=max(tomllib.loads((H/'control/slicing_validation.toml').read_text())['errors']);assert sliceerr<1e-12
reuse=tomllib.loads((H/'control/native_reuse_validation.toml').read_text())['checks']
assert len(reuse)==6
reuse_error=max(max(v['errors']) for v in reuse);assert reuse_error<1e-10
note=H.parent.parent/'docs/13_大きいtubeで電荷準位から二量体化を読む.md';s=note.read_text()
assert '\\(' not in s and '\\[' not in s
links=re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)',s)
for link in links:
    if not link.startswith('http'):assert (note.parent/link).exists(),link
formulas=re.findall(r'\$\$(.*?)\$\$|(?<!\$)\$([^\n$]+)\$(?!\$)',s,re.S)
(H/'control/formulas.json').write_text(json.dumps([dict(tex=a or b,display=bool(a)) for a,b in formulas],ensure_ascii=False,indent=2)+'\n')
katex=json.loads(subprocess.check_output(['node',str(H/'check_math.cjs')],text=True))
copies=json.loads((H/'control/local_cft_copies.json').read_text())
for c in copies:assert sha(ROOT/c['source'])==sha(ROOT/c['destination'])==c['sha256']
old_copies=json.loads((H/'control/local_cft_copies.json.before_early').read_text())
for c in old_copies:
    p=ROOT/c['destination'];assert sha(p)==c['sha256']
    for original in tomllib.loads(p.read_text())['source_files']:
        assert sha(ROOT/original['path'])==original['sha256']
result=dict(measurements=len(wide),charged_levels=3*len(wide),initial_vectors_per_sector=2,
    max_residual=max(r['residual'] for r in wide),max_start_difference=max(r['agreement'] for r in wide),
    independent_geometry_error=geoerr,slicing_error=sliceerr,native_reuse_error=reuse_error,same_checkpoint_pairs=len(wide),local_cft_copies=len(copies),
    checked_links=len(links),math_formulas=len(formulas),katex=katex,new_RG_tensors=0,drive_uploaded=False)
(H/'control/validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
