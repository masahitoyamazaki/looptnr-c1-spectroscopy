"""Known marginal (J Jbar) injection: O(2), chi = 40, runs at K and K +- delta (delta = 1e-3), read with the K_n
estimator of ../../cpu_baseline_20261106/analysis/drift.py (imported, not modified).

Usage:  python3 jj.py          (reads ../data/JJ_K<K>_chi40/)
A shift of K is the exactly marginal J Jbar deformation.  For each base K we compute
    dK_n(n) = K_n(K + delta) - K_n(K)
in the window n = 8 .. n_max (whole run at K = 2.5; at K = 1.25 the floor window, before the dimerization grows)
and report its mean over delta (should be 1 if the estimator reads the injected coupling back) and its slope per
step over delta (should be 0 if the injected coupling is neither amplified nor damped, i.e. exactly marginal).
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "main", "analysis"))
from drift import analyze   # noqa: E402

DATA = os.path.join(HERE, "..", "data")
WINDOW = {2.5: (8, 70), 1.25: (8, 24)}
for K0, (n1, n2) in WINDOW.items():
    base = {r["n"]: r for r in analyze(os.path.join(DATA, f"JJ_K{K0}_chi40"), K0, 40)[1]}
    print(f"K = {K0}: window n = {n1}-{n2}; base run dK_n/dn = "
          f"{np.polyfit([n for n in base if n1 <= n <= n2], [base[n]['K_n'] for n in base if n1 <= n <= n2], 1)[0]:+.2e}")
    for d in (+1e-3, -1e-3):
        K = round(K0 + d, 3)
        run = {r["n"]: r for r in analyze(os.path.join(DATA, f"JJ_K{K}_chi40"), K, 40)[1]}
        ns = [n for n in sorted(run) if n in base and n1 <= n <= n2]
        dk = np.array([run[n]["K_n"] - base[n]["K_n"] for n in ns])
        de = np.array([run[n]["eta"] - base[n]["eta"] for n in ns])
        slope = np.polyfit(ns, dk, 1)[0]
        print(f"   K = {K}: <dK_n>/delta = {dk.mean() / d:.4f} (min {dk.min() / d:.4f}, max {dk.max() / d:.4f}), "
              f"slope/delta = {slope / d:+.2e} per step, <d eta> = {de.mean():+.1e}, steps {len(ns)}")

# Antisymmetric combination a_n = [K_n(K+delta) - K_n(K-delta)] / 2 delta (numbers quoted in the paper, Sec. VII):
# mean with standard error, and slope per step with the standard error of the fitted slope.
print("antisymmetric combination:")
for K0, (n1, n2) in WINDOW.items():
    d = 1e-3
    R = {K: {r["n"]: r for r in analyze(os.path.join(DATA, f"JJ_K{K}_chi40"), K, 40)[1]}
         for K in (K0, round(K0 + d, 3), round(K0 - d, 3))}
    ns = [n for n in sorted(R[K0]) if n1 <= n <= n2 and all(n in R[k] for k in R)]
    a = np.array([(R[round(K0 + d, 3)][n]["K_n"] - R[round(K0 - d, 3)][n]["K_n"]) / (2 * d) for n in ns])
    sl, ic = np.polyfit(ns, a, 1)
    res = a - np.polyval([sl, ic], ns)
    se = res.std(ddof=2) / np.sqrt(((np.array(ns) - np.mean(ns)) ** 2).sum())
    print(f"   K = {K0}: n = {ns[0]}-{ns[-1]} ({len(ns)} steps): <a> = {a.mean():.4f} +- {a.std(ddof=1) / np.sqrt(len(a)):.4f}, "
          f"slope = {sl:+.2e} +- {se:.1e} per step, y = 2 slope / ln 2 = {2 * sl / np.log(2):+.4f}")
