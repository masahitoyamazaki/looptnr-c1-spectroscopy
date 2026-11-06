from pathlib import Path
import csv, json, math, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
import os
try: import tomllib as tomli
except ModuleNotFoundError: import tomli
from scipy.optimize import minimize_scalar
from bundle import check_csv,finish
from plot import plot_legacy_appendix
CS=[.5,1.,1.4,1.7,1.9,1.98]
LB=math.log(2)/2
def exact_k(c):
    return math.pi/(2*math.acos(1-c*c/2))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    paths = list((ROOT/'data/cft').rglob('*.toml'))
    index, provenance = {}, []
    for p in sorted(paths):
        d = tomli.loads(p.read_text()); o = d['observables'].copy()
        tube = 'small' if p.parent.name == 'measurements' else 'wide'
        o.update(tube=tube, record=str(p.relative_to(ROOT)))
        if o['rg_step'] < 15:
            continue
        k = exact_k(o['vertex_c']); o['K_exact'] = k
        for q in range(1, 4):
            plus = o.get(f'x{q}', np.nan)
            minus = o.get(f'xm{q}', np.nan) if o['symmetry'] == 'U1Irrep' else plus
            o[f'E{q}'] = (plus+minus)/2
            o[f'O{q}'] = (plus-minus)/2
            o[f'exact{q}'] = q*q/(4*k)
            o[f'dE{q}'] = o[f'E{q}']-o[f'exact{q}']
        odd = np.array([o[f'O{q}'] for q in range(1,4)])
        charges = np.arange(1,4)
        o['B'] = float(odd @ charges / 14)
        o['odd_nonlinearity'] = (float(np.linalg.norm(odd-o['B']*charges)/np.linalg.norm(odd)) if np.linalg.norm(odd)>0 else 0.) if np.all(np.isfinite(odd)) else np.nan
        o['R2'] = o['E2']/4-o['E1']; o['R3'] = o['E3']/9-o['E1']
        o['G2'] = o['E2']/(4*o['E1'])-1 if o['E1'] else np.nan
        o['G3'] = o['E3']/(9*o['E1'])-1 if o['E1'] else np.nan
        aspect = .5 if tube=='wide' else 1.
        o['fixed_metric_E1'] = o['E1']*o.get('tau_tm_imag',np.nan)/aspect
        leading = [v for v in d.get('spectra',[]) if v['rank']==1]
        relres = [v.get('residual_relative', np.nan) for v in leading]
        o['max_leading_relative_residual'] = max(relres, default=np.nan)
        o['all_requested_converged'] = all(v.get('converged',0)>=v.get('requested',6) for v in d.get('sectors',[]))
        o['warnings'] = str(d.get('warnings',''))
        key = (o['symmetry'], o['vertex_c'], o['rg_step'], tube)
        if key in index:
            raise RuntimeError(f'Duplicate measurement {key}')
        index[key] = o
        provenance.append({'record':o['record'], 'sha256':sha(p), 'checkpoint_sha256':o.get('checkpoint_sha256','')})
    return index, provenance


def series(index, sym, c, tube, lo=15, hi=70):
    return [index[sym,c,n,tube] for n in range(lo,hi+1) if (sym,c,n,tube) in index]


def oddfit(data):
    n=np.array([o['rg_step'] for o in data],float); y=np.array([o['O1'] for o in data]); t=n-n[-1]
    def fit(rate):
        X=np.exp(rate*t)[:,None]*np.array([np.cos(3*np.pi*n/4),np.sin(3*np.pi*n/4)]).T
        coeff=np.linalg.lstsq(X,y,rcond=None)[0]; pred=X@coeff
        return np.sum((y-pred)**2),coeff,pred
    opt=minimize_scalar(lambda r:fit(r)[0],bounds=(.1,.7),method='bounded',options={'xatol':1e-10})
    loss,coeff,pred=fit(opt.x)
    return dict(rate=float(opt.x),power=float(opt.x/LB),amplitude_end=float(np.linalg.norm(coeff)),
                cos_coeff=float(coeff[0]),sin_coeff=float(coeff[1]),rmse=float(np.sqrt(loss/len(n))),
                normalized_rmse=float(np.sqrt(loss/np.sum(y*y))),start=int(n[0]),end=int(n[-1]),npoints=len(n)),pred


def odd_fits(index):
    rows=[]; sens=[]
    for c in CS:
        data=series(index,'U1Irrep',c,'wide'); good=[]
        for o in data:
            if (o['status']!='ok' or abs(o['O1'])/o['exact1']>.25 or abs(o['dE1'])/o['exact1']>.1 or abs(o['central_charge']-1)>.2):break
            good.append(o)
        if len(good)<10:continue
        end=good[-1]['rg_step'];lo=max(18,end-11)
        d=series(index,'U1Irrep',c,'wide',lo,end)
        if len(d)!=end-lo+1:continue
        f,_=oddfit(d)
        small=series(index,'U1Irrep',c,'small',lo,end)
        # Direct same-step ratio projection, robust at oscillation nodes.
        bs=np.array([o['O1'] for o in small]); bw=np.array([o['O1'] for o in d]); ratio=float(bs@bw/(bs@bs))
        charge_res=float(np.sqrt(sum(sum((o[f'O{q}']-q*o['B'])**2 for q in range(1,4)) for o in d)/sum(sum(o[f'O{q}']**2 for q in range(1,4)) for o in d)))
        for short in [0,1,2]:
            for length in [8,10,12]:
                dd=series(index,'U1Irrep',c,'wide',max(18,end-short-length+1),end-short)
                if len(dd)<8:continue
                ff,_=oddfit(dd);sens.append(dict(vertex_c=c,**ff))
        pp=[v['power'] for v in sens if v['vertex_c']==c]
        rows.append(dict(vertex_c=c,K_exact=exact_k(c),**f,sensitivity_min=min(pp),sensitivity_max=max(pp),
                         tube_ratio=ratio,tube_ratio_relative_residual=float(np.linalg.norm(bw-ratio*bs)/np.linalg.norm(bw)),
                         charge_linearity_residual=charge_res))
    return rows,sens


def main():
    index,provenance=load()
    rows,sensitivity=odd_fits(index)
    for name,values in [('odd_current_fits.csv',rows),('odd_fit_sensitivity.csv',sensitivity)]:
        with (ROOT/'data/processed'/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
        check_csv(name)
    audit=[]
    for (sym,c,n,tube),point in sorted(index.items()):
        primary=next(r for r in rows if r['vertex_c']==c)
        in_primary=primary['start']<=n<=primary['end']
        in_sensitivity=tube=='wide' and any(r['vertex_c']==c and r['start']<=n<=r['end'] for r in sensitivity)
        audit.append(dict(symmetry=sym,vertex_c=c,rg_step=n,tube=tube,
                          status=point['status'],record=point['record'],
                          in_primary=in_primary,in_sensitivity=in_sensitivity,
                          role='primary' if in_primary else 'sensitivity_only' if in_sensitivity else 'not_used'))
    with (ROOT/'data/processed/selection_audit.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(audit[0]));writer.writeheader();writer.writerows(audit)
    plot_legacy_appendix()
    finish(dict(primary_fits=len(rows),measurement_records=len(index),omega_fixed_degrees=135))
