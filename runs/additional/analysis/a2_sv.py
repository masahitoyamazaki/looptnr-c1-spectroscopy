"""A2: singular-value spectra at the chi cut (../sv/out/*.csv.gz from ../sv/sv_spectra.jl) versus the seed g_d^2(12).

Usage:  python3 a2_sv.py
O(2): for every chi = 24-48 (K = 1, 1.5) and steps 5-10, discarded weight and gap at the cut on the 8 loop bonds,
correlated with g_d^2(12) of b1_b2.csv (run B2 and baseline).  C only (B7): whether the cut splits an exactly
degenerate C-even/C-odd pair (a U(1) doublet), the mechanism discussed in main_en.tex Sec. V.C.
"""
import csv, gzip, math, os
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SV = os.path.join(HERE, "..", "sv", "out")
A = os.path.join(HERE, "b1_b2.csv")
seed={}
for r in csv.DictReader(open(A)):
    if r["set"]=="B2": seed[(float(r["K"]),int(r["chi"]))]=(float(r["g2_n12"]),int(r["n_e"]))
def cuts(path):
    sp=defaultdict(list)
    for r in csv.DictReader(gzip.open(path, "rt")):
        sp[(int(r["step"]),int(r["bond"]))].append((float(r["value"]),int(r["mult"]),r["sector"]))
    return sp
def metrics(v,chi):
    v=sorted(v,key=lambda t:-t[0]); kept=0; i=0
    while i<len(v) and kept+v[i][1]<=chi: kept+=v[i][1]; i+=1
    tot=sum(m*x*x for x,m,_ in v); disc=sum(m*x*x for x,m,_ in v[i:])
    gap=v[i][0]/v[i-1][0] if i<len(v) else 0.0
    split = (i<len(v) and abs(v[i][0]-v[i-1][0])/v[i-1][0]<1e-6)
    return kept, disc/tot, gap, split, (v[i][2] if i<len(v) else "")
out=[]
for K in (1.0,1.5):
    print(f"\n## K={K}   chi | g2(12) n_e | step5: kept eps_max gap_max | step6 | step7 | step8")
    for chi in range(24,49):
        sp=cuts(os.path.join(SV, f"CU1_K{K}_chi{chi}.csv.gz"))
        row=[]; feats={}
        for n in (5,6,7,8,9,10):
            ms=[metrics(sp[(n,b)],chi) for b in range(1,9) if (n,b) in sp]
            if not ms: continue
            kept=min(m[0] for m in ms); eps=max(m[1] for m in ms); gap=max(m[2] for m in ms); nsplit=sum(m[3] for m in ms)
            feats[n]=(kept,eps,gap,nsplit)
        g2,ne=seed.get((K,chi),(float('nan'),0))
        out.append((K,chi,g2,ne,feats))
        s=" | ".join(f"{feats[n][0]:2d} {feats[n][1]:.1e} {feats[n][2]:.3f}{'*' if feats[n][3] else ''}" for n in (5,6,7,8) if n in feats)
        print(f"  {chi:2d} | {g2:.1e} {ne:2d} | {s}")
    # correlations
    for n in (5,6,7,8,9,10):
        x=[];y=[];z=[]
        for (KK,chi,g2,ne,f) in out:
            if KK==K and n in f and g2>0: x.append(math.log10(f[n][1])); z.append(f[n][2]); y.append(math.log10(g2))
        print(f"  step {n}: corr(log g2, log eps) = {np.corrcoef(x,y)[0,1]:+.2f}   corr(log g2, gap) = {np.corrcoef(z,y)[0,1]:+.2f}")
    x=[];y=[]
    for (KK,chi,g2,ne,f) in out:
        if KK==K and g2>0: x.append(math.log10(sum(f[n][1] for n in (5,6,7,8,9,10) if n in f))); y.append(math.log10(g2))
    print(f"  sum eps steps5-10: corr = {np.corrcoef(x,y)[0,1]:+.2f}")

print("\n## near-degenerate splits at the cut (gap > thr), summed over bonds and steps 5..N")
for thr in (0.999, 0.99, 0.95):
  for N in (6, 8, 10):
    for K in (1.0, 1.5):
        xs=[];ys=[]
        for chi in range(24,49):
            sp=cuts(os.path.join(SV, f"CU1_K{K}_chi{chi}.csv.gz"))
            cnt=sum(1 for (n,b),v in sp.items() if 5<=n<=N and metrics(v,chi)[2]>thr)
            g2,ne=seed[(K,chi)]; xs.append(cnt); ys.append(math.log10(g2))
        print(f" thr={thr} N={N} K={K}: corr={np.corrcoef(xs,ys)[0,1]:+.2f}  counts={xs}")

print("C-only (B7): at each step, bonds whose chi cut splits a pair (s_last, s_next) with rel. diff < tol, by sector of the pair")
for K in ("0.75","1.0","1.25","1.5","1.75","2.5"):
  for chi in (24,32,40):
    sp=cuts(os.path.join(SV, f"C_K{K}_chi{chi}.csv.gz")); line=[]
    for n in range(5,10):
        cnt=0; cross=0; rd=[]
        for b in range(1,9):
            v=sorted(sp.get((n,b),[]),key=lambda t:-t[0])
            if len(v)<=chi: continue
            a,c=v[chi-1],v[chi]
            d=abs(a[0]-c[0])/a[0]
            rd.append(d)
            if d<1e-6:
                cnt+=1; cross+= (a[2]!=c[2])
        line.append(f"n{n}: {cnt}/{len(rd)} split, {cross} even|odd, min rel {min(rd) if rd else float('nan'):.0e}")
    print(f"K={K:4s} chi={chi}: " + " ; ".join(line))
