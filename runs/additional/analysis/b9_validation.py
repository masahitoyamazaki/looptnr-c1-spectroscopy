"""B9: validation of the compensation on levels not used for the calibration (all assigned levels except the
lowest charged one).  Usage: python3 b9_validation.py   (reads ../data/B9_*)
"""
import sys, glob, os, re, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cft_spectrum import predicted, load_run, comparable_k
def E_series(d, K, exclude=("CU1(1,2)",0)):
    pred=predicted(K); steps,lv=load_run(d); out={}
    for n in sorted(steps):
        errs=[]; per={}
        for sec,xs in lv[n].items():
            xs=sorted(xs); p=pred[sec]; k=min(comparable_k(p,len(xs)),len(xs))
            idx=[i for i in range(k) if not (sec==exclude[0] and i==exclude[1])]
            errs+= [xs[i]-p[i] for i in idx]
        out[n]=(float(np.sqrt(np.mean(np.square(errs)))) if errs else np.nan, steps[n]["c"], len(errs))
    return out
def summary(E, tol=1e-2, ctol=5e-3):
    good=[n for n,(e,c,_) in E.items() if n>=6 and e<tol and abs(c-1)<ctol]
    # longest run
    best=(0,None,None); cur=0; st=None; prev=None
    for n in sorted(good):
        if prev is not None and n==prev+1: cur+=1
        else: cur=1; st=n
        if cur>best[0]: best=(cur,st,n)
        prev=n
    L,a,b=best
    win=[E[n][0] for n in range(a,b+1)] if a else []
    brk=next((n for n in sorted(E) if a and n>b and E[n][0]>0.05),None)
    return L,(a,b),(np.median(win) if win else np.nan),brk
cases={(0.75,32):"ab+2.50e-06",(0.75,40):"ab+1.50e-06",(1.0,32):"ab+6e-06",(1.0,40):"ab+4e-06",(1.25,32):"ab+9.00e-06",(1.25,40):"ab+6.00e-06",
       (1.5,32):"ab+3.5e-05",(1.5,40):"ab+2e-05",(1.75,32):"ab+1.10e-04",(1.75,40):"ab+1.25e-04"}   # run closest to h* (b9.py)
print("validation set: all comparable levels except the lowest charged level (used for g_d)")
print(f"{'K':>4} {'chi':>3} {'h':>8} | plateau(E<1e-2,|c-1|<5e-3)  median E   step E>0.05 | nlev")
for (K,chi),tg in cases.items():
    for h,tag in ((0,"ab0"),(float(tg[2:]),tg)):
        d=os.path.join(HERE, "..", "data", f"B9_K{K}_chi{chi}_{tag}")
        if not os.path.isdir(d): print("missing",d); continue
        E=E_series(d,K); L,w,med,brk=summary(E)
        nl=int(np.median([v[2] for v in E.values()]))
        print(f"{K:4} {chi:3} {h:8.1e} | {L:3d} steps {str(w):>10}   {med:.2e}   {brk} | {nl}")
