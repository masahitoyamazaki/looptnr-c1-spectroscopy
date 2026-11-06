"""Joint fit of the mean and splitting of the charge-(+-1) levels with only C imposed (main_en.tex Sec. V.C).

Usage:  python3 b7_joint_fit.py
Imports (does not modify) Yamamoto/finite_bonddim/src/conly_cpt_fit.py (report 7): coefficients(), inputs(), weak().
Compares, on Yamamoto's chi=16 scan and on runs B7 (../data), three models with the theoretical rates:
d^2 + V_{0,+-2}; + second-order V_{0,+-1} (report 7); + free second-order V_{0,+-2} in M only.
Then extrapolates the fitted V_{0,+-1} and V_{0,+-2} couplings back to n = 5.

Original note: Refit Yamamoto's C-only chi=16 scan (report 7) with W2^2 (second order of V_{0,+-2}, free sign, rate r2^2, in M only)
instead of W1^2.  Uses Yamamoto's coefficients()/inputs()/weak() unchanged (imported, not modified)."""
import sys, types, math, numpy as np
sys.modules['usb_stage3_ingest']=types.SimpleNamespace(memory=lambda: {})
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "external", "yamamoto_conly_scan", "finite_bonddim", "src"))
import conly_cpt_fit as Y
from scipy.optimize import lsq_linear
def fit(part,c,model):
    n=np.array([r['n'] for r in part],float); t=n-n.max(); z=np.zeros(len(t)); one=np.ones(len(t)); xs=1/(4*c['K'])
    y=np.r_[[r['M']-xs for r in part],[r['S'] for r in part]]
    cols=[np.r_[one,z],np.r_[t,z],np.r_[z,one]]; names=['M0','Md','S0']; lo=[-np.inf]*3
    if c['d_included']: cols.append(np.r_[c['alpha_d']*c['r_d']**t,z]); names.append('gd2'); lo.append(0)
    cols.append(np.r_[z,math.sqrt(2)*math.pi*c['r_2']**t]); names.append('g2'); lo.append(-np.inf)
    if model=='W1sq': cols.append(np.r_[c['beta_mean']*c['r_1']**t,c['beta_split']*c['r_1']**t]); names.append('g1sq'); lo.append(0)
    if model=='W2sq': cols.append(np.r_[(c['r_2']**2)**t,z]); names.append('w2sq'); lo.append(-np.inf)
    A=np.array(cols).T; nr=np.linalg.norm(A,axis=0); f=lsq_linear(A/nr,y,bounds=(lo,[np.inf]*len(lo)),tol=1e-13)
    v=f.x/nr; pred=A@v; N=len(part); dM=pred[:N]; S=pred[N:]
    E=xs+dM+S; O=xs+dM-S
    err=np.r_[E-[r['x_Ceven'] for r in part],O-[r['x_Codd'] for r in part]]
    return float(np.sqrt(np.mean(err**2))/xs)
print(" K     window | RMSE%: d2+W2 | +W1^2 (report 7) | +W2^2 (free, M only)")
for (src,run),rr in sorted(Y.inputs().items(), key=lambda kv: kv[1][0]['K']):
    if src!='Yamamoto': continue
    K=rr[0]['K']; c=Y.coefficients(K); part=Y.weak(rr,8)
    if len(part)<8: continue
    part=part[-10:]
    a,b,d=(fit(part,c,m)*100 for m in ('none','W1sq','W2sq'))
    print(f"{K:5} {part[0]['n']:3d}-{part[-1]['n']:<3d} | {a:8.4f} | {b:8.4f} | {d:8.4f}")

# --- same models on runs B7 (chi = 24, 32, 40) ---
import csv, gzip, os
B7=os.path.join(HERE, '..', 'data')
def b7_rows(K,chi):
    d=os.path.join(B7,f"B7_C_K{K}_chi{chi}"); lev={}; c={}
    for r in csv.DictReader(gzip.open(d+"/levels.csv.gz","rt")): lev.setdefault(int(r["step"]),[]).append(r)
    for r in csv.DictReader(gzip.open(d+"/steps.csv.gz","rt")): c[int(r["step"])]=float(r["central_charge"])
    xs=1/(4*K); out=[]
    for n in sorted(lev):
        ev=sorted(float(r["x"]) for r in lev[n] if "(0)" in r["sector"])[1:]
        od=sorted(float(r["x"]) for r in lev[n] if "(1)" in r["sector"])
        e=min(ev,key=lambda x:abs(x-xs)); o=min(od,key=lambda x:abs(x-xs))
        out.append(dict(n=n,K=K,xstar=xs,central_charge=c[n],x_Ceven=e,x_Codd=o,M=(e+o)/2,S=(e-o)/2,valid=True))
    return out
print("\nB7 (this work):  K chi window | d2+W2 | +W1^2 | +W2^2")
for K in (0.75,1.0,1.25,1.5,1.75,2.5):
    for chi in (24,32,40):
        rr=b7_rows(K,chi); c=Y.coefficients(K); part=Y.weak(rr,8)
        if len(part)<8: print(K,chi,"short"); continue
        part=part[-10:]
        a,b,d=(fit(part,c,m)*100 for m in ('none','W1sq','W2sq'))
        print(f"{K:5} {chi:3d} {part[0]['n']:3d}-{part[-1]['n']:<3d} | {a:8.4f} | {b:8.4f} | {d:8.4f}")

# --- extrapolate the fitted couplings back to n = 5 (first truncation at chi=40) ---
def params(part,c):
    n=np.array([r['n'] for r in part],float); t=n-n.max(); z=np.zeros(len(t)); one=np.ones(len(t)); xs=1/(4*c['K'])
    y=np.r_[[r['M']-xs for r in part],[r['S'] for r in part]]
    cols=[np.r_[one,z],np.r_[t,z],np.r_[z,one]]; names=['M0','Md','S0']; lo=[-np.inf]*3
    if c['d_included']: cols.append(np.r_[c['alpha_d']*c['r_d']**t,z]); names.append('gd2'); lo.append(0)
    cols.append(np.r_[z,math.sqrt(2)*math.pi*c['r_2']**t]); names.append('g2'); lo.append(-np.inf)
    cols.append(np.r_[c['beta_mean']*c['r_1']**t,c['beta_split']*c['r_1']**t]); names.append('g1sq'); lo.append(0)
    A=np.array(cols).T; nr=np.linalg.norm(A,axis=0); f=lsq_linear(A/nr,y,bounds=(lo,[np.inf]*len(lo)),tol=1e-13)
    return dict(zip(names,f.x/nr)), n.max()
print("\nback-extrapolation to n=5:  K chi | n_e | g1^2(n_e) -> |g1|(5) | |g2|(n_e) -> |g2|(5)")
for K,chi in ((1.0,40),(1.25,24),(1.5,24),(1.5,40),(1.75,24),(2.5,24),(1.75,40),(1.0,32)):
    rr=b7_rows(K,chi); c=Y.coefficients(K); part=Y.weak(rr,8)[-10:]
    p,ne=params(part,c)
    g1e=math.sqrt(max(p['g1sq'],0)); g2e=abs(p['g2'])
    s1=c['r_1']**0.5; s2=c['r_2']   # per-step growth of |g1| and |g2|
    print(f"  {K:4} {chi:3d} | {int(ne):3d} | {p['g1sq']:.1e} -> {g1e/s1**(ne-5):.1e} | {g2e:.1e} -> {g2e/s2**(ne-5):.1e}")
