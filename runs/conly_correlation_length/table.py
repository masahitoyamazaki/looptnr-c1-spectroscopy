"""Table for paper 1b: C-only runs, lowest C-odd gap vs the first-breather prediction from the V_{0,+-2} seed."""
import os, sys, math
sys.argv = [sys.argv[0]]
import xi_conly as X
import numpy as np, glob
# residual of the joint fit without / with the V_{0,+-1} term (followup_B b7_joint_fit.py output, 2026-11-12)
JF = {(0.75,24):(0.0109,0.0109),(0.75,32):(0.0148,0.0148),(0.75,40):(0.0048,0.0048),(1.0,24):(0.0172,0.0170),
      (1.0,32):(0.0027,0.0027),(1.0,40):(0.0472,0.0040),(1.25,24):(0.2301,0.0058),(1.25,32):(0.0079,0.0065),
      (1.25,40):(0.0087,0.0053),(1.5,24):(0.1802,0.0064),(1.5,32):(0.0093,0.0073),(1.5,40):(0.6337,0.0096),
      (1.75,24):(0.5963,0.0039),(1.75,32):(0.0169,0.0064),(1.75,40):(0.0198,0.0055),(2.5,24):(0.3540,0.0305),
      (2.5,32):(0.0397,0.0211),(2.5,40):(0.0462,0.0063)}
rows = []
for d in sorted(glob.glob(os.path.join(X.B7, "B7_C_K*_chi*"))):
    name = os.path.basename(d); K = float(name.split("_K")[1].split("_")[0]); chi = int(name.split("chi")[1])
    if K <= 1: continue
    lev, st = X.load(d); s = X.series(lev, K)
    x = 1/K; p = x/(2-x); r = 2**(1-1/(2*K)); ref = 1/(4*K)
    D = {}
    for n, (ev, od) in s.items():
        if len(ev) < 2 or not od: continue
        D[n] = min(ev[1:], key=lambda v: abs(v-ref)) - min(od, key=lambda v: abs(v-ref))
    win = []
    for n in sorted(D):
        if n < 10: continue
        if 1e-5 < 4*K*abs(D[n]) < 1e-2: win.append(n)
        elif win: break
    if len(win) < 4: continue
    n2 = win[-1]; t = np.array([r**(n-n2) for n in win]); y = np.array([D[n] for n in win]); A = float(t@y/(t@t))
    g12 = abs(A)*r**(12-n2)/(2*math.sqrt(2)*math.pi); L12 = 2*math.sqrt(2)*math.sqrt(2)**12
    M = (2*math.pi/L12)*(g12/(math.sqrt(2)*X.kappa(p)))**(1/(2-x))
    nl = max(s); L = 2*math.sqrt(2)*math.sqrt(2)**nl
    odd = [2*math.pi*v/L/M for v in s[nl][1]]; m1 = min(v for v in odd if v > 0.3)
    nb = 2*math.sin(math.pi*p/2) if p < 1 else 2.0
    a, b = JF[(K, chi)]
    rows.append((K, chi, f"{win[0]}-{n2}", g12, 1/M, m1, nb, a/b))
print(" K   chi  window  g02(12)   xi_pred   m_odd/M  pred   m/pred  V01 fit gain")
for K, chi, w, g, xi, m, nb, gain in rows:
    print(f"{K:4} {chi:4d}  {w:6s} {g:9.2e} {xi:9.2e}  {m:7.3f}  {nb:5.3f}  {m/nb:6.3f}  {gain:6.1f}")
