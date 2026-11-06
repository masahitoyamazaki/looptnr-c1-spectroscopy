"""Free-ratio test of the fast component in the C-only runs B7 (main_en.tex Sec. V.C): fitted S/M ratio of a term
growing as 2^{2-1/(4K)} versus the V_{0,+-1} prediction b_S/b_M.  Usage: python3 b7_v01_ratio.py"""
import sys, os, math, numpy as np, csv, gzip, types
sys.modules['usb_stage3_ingest']=types.SimpleNamespace(memory=lambda: {})
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "external", "yamamoto_conly_scan", "finite_bonddim", "src")); import conly_cpt_fit as Y
from scipy.optimize import lsq_linear
B7=os.path.join(HERE, "..", "data")
def b7_rows(K,chi):
    d=os.path.join(B7,f"B7_C_K{K}_chi{chi}"); lev={}; cc={}
    for r in csv.DictReader(gzip.open(d+"/levels.csv.gz","rt")): lev.setdefault(int(r["step"]),[]).append(r)
    for r in csv.DictReader(gzip.open(d+"/steps.csv.gz","rt")): cc[int(r["step"])]=float(r["central_charge"])
    xs=1/(4*K); out=[]
    for n in sorted(lev):
        ev=sorted(float(r["x"]) for r in lev[n] if "(0)" in r["sector"])[1:]
        od=sorted(float(r["x"]) for r in lev[n] if "(1)" in r["sector"])
        e=min(ev,key=lambda x:abs(x-xs)); o=min(od,key=lambda x:abs(x-xs))
        out.append(dict(n=n,K=K,xstar=xs,central_charge=cc[n],x_Ceven=e,x_Codd=o,M=(e+o)/2,S=(e-o)/2,valid=True))
    return out
print(" K   chi | fast-term ratio S/M (free) | theory b_S/b_M | RMSE free vs fixed-ratio (%)")
for K,chi in ((1.0,40),(1.25,24),(1.5,24),(1.5,40),(1.75,24),(2.5,24)):
    rr=b7_rows(K,chi); c=Y.coefficients(K); part=Y.weak(rr,8)[-10:]
    n=np.array([r['n'] for r in part],float); t=n-n.max(); z=np.zeros(len(t)); one=np.ones(len(t)); xs=1/(4*K)
    y=np.r_[[r['M']-xs for r in part],[r['S'] for r in part]]
    cols=[np.r_[one,z],np.r_[t,z],np.r_[z,one]]
    if c['d_included']: cols.append(np.r_[c['alpha_d']*c['r_d']**t,z])
    cols.append(np.r_[z,math.sqrt(2)*math.pi*c['r_2']**t])
    cols.append(np.r_[c['r_1']**t,z]); cols.append(np.r_[z,c['r_1']**t])     # free amplitudes in M and S
    A=np.array(cols).T; nr=np.linalg.norm(A,axis=0); v=np.linalg.lstsq(A/nr,y,rcond=None)[0]/nr
    aM,aS=v[-2],v[-1]
    pred=A@v; rm=np.sqrt(np.mean((pred-y)**2))/xs*100
    print(f" {K:4} {chi:3d} | {aS/aM:+.3f} (M amp {aM:+.1e}) | {c['beta_split']/c['beta_mean']:+.3f} | {rm:.4f}")
