"""Period-four component of the common scale error eta(n) (Sec. V B; referee item on T, Tbar).

T and Tbar (spin +-2, exactly marginal) are allowed because C4 is not imposed; their coupling rotates by 270 degrees per
LoopTNR step and enters eta with period four in n.  For each O(2) run we fit eta(n), n >= 12, with
    a + b (n - <n>) + c cos(pi n / 2) + d sin(pi n / 2) + e (-1)^n
and print the period-four amplitude sqrt(c^2 + d^2), the period-two amplitude |e|, and the rms residual.
Only K = 2.5 (no growing perturbation, whole run usable) is meaningful; for K < 2 the fit range includes the growth.
Usage:  python3 eta_period4.py        (reads drift_steps.csv written by drift.py)
"""
import csv, os, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "drift_steps.csv"))))
g = collections.defaultdict(dict)
for r in rows:
    g[(float(r["K"]), int(r["chi"]))][int(r["n"])] = r
for (K, chi) in sorted(g):
    if K < 2 or chi < 24:
        continue
    d = g[(K, chi)]; ns = sorted(n for n in d if n >= 12)
    n = np.array(ns, float); y = np.array([float(d[m]["eta"]) for m in ns])
    M = np.vstack([np.ones_like(n), n - n.mean(), np.cos(np.pi * n / 2), np.sin(np.pi * n / 2), (-1.0) ** n]).T
    c, *_ = np.linalg.lstsq(M, y, rcond=None)
    res = y - M @ c
    print(f"K={K} chi={chi} n={ns[0]}-{ns[-1]}: period-4 amplitude {np.hypot(c[2], c[3]):.1e}, "
          f"period-2 {abs(c[4]):.1e}, residual rms {res.std():.1e}")
