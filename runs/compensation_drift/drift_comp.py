"""Payoff of compensation: direct drift of K_n in the floor window, uncompensated vs compensated (paper 1b).

K_n and eta from drift.py (main/analysis).  Floor window: 12 <= n <= n_e - ln(1000)/ln(2^{2-K}),
i.e. while the dimerization term (|X| = 0.1 at n_e) is below 1e-4; the same rule for all runs.
Compensated run: the B9 run closest to mu* of Table I of paper 1b.
"""
import math, os, sys, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "main", "analysis"))
import drift  # noqa
BASE = os.path.join(HERE, "..", "main", "data")
B9 = os.path.join(HERE, "..", "additional", "data")
MUSTAR = {(0.75,32):2.6e-6,(0.75,40):1.5e-6,(1.0,32):5.4e-6,(1.0,40):4.2e-6,(1.25,32):7.2e-6,(1.25,40):5.9e-6,
          (1.5,32):3.6e-5,(1.5,40):2.2e-5,(1.75,32):1.3e-4,(1.75,40):1.2e-4}

def fitdrift(steps, K):
    X = {r["n"]: r["X"] for r in steps}
    bad = [n for n in sorted(X) if n >= 6 and abs(X[n]) > 0.1]
    ne = (bad[0] - 1) if bad else max(X)
    nmax = ne - math.log(1000) / math.log(2 ** (2 - K))
    fl = [r for r in steps if 12 <= r["n"] <= nmax]
    if len(fl) < 4: return ne, None
    n = np.array([r["n"] for r in fl], float); k = np.array([r["K_n"] for r in fl])
    A = np.vstack([n, np.ones_like(n)]).T
    coef, res, *_ = np.linalg.lstsq(A, k, rcond=None)
    dof = len(n) - 2
    s2 = float(np.sum((A @ coef - k) ** 2)) / max(dof, 1)
    se = math.sqrt(s2 / np.sum((n - n.mean()) ** 2))
    return ne, dict(n1=int(n[0]), n2=int(n[-1]), slope=coef[0], se=se)

print(" K  chi |   mu=0: n_e floor    dK/dn (s.e.)        | compensated mu: n_e floor    dK/dn (s.e.)")
for (K, chi), ms in sorted(MUSTAR.items()):
    d0 = os.path.join(BASE, f"CU1_K{K:.4f}_chi{chi}")
    cands = [c for pat in (f"B9_K{K:g}_chi{chi}_ab*", f"B9_K{K:.1f}_chi{chi}_ab*", f"B9*_K{K:g}_chi{chi}_ab*") for c in glob.glob(os.path.join(B9, pat))]
    def mu(p):
        try: return float(p.split("_ab")[1])
        except ValueError: return math.inf
    dc = min(cands, key=lambda p: abs(mu(p) - ms)) if cands else None
    out = []
    for d in (d0, dc):
        if d is None or not os.path.isdir(d): out.append("   (missing)"); continue
        _, steps = drift.analyze(d, K, chi)
        ne, f = fitdrift(steps, K)
        out.append(f"{ne:3d} {f['n1']:2d}-{f['n2']:<2d} {f['slope']:+.2e} ({f['se']:.1e})" if f else f"{ne:3d} no floor")
    print(f"{K:4} {chi:3d} | {out[0]:36s} | {(mu(dc) if dc else float("nan")):.2e}: {out[1]}")
