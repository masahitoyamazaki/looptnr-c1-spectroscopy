"""Ratio (first-order twisted splitting)^2 / (second-order shift of the charged level) from the six-vertex ED
(ratio_ed.csv, mu = 0 and 1e-3), compared with 8 pi^2/alpha_{0,1} (Paper 1b, App. C, ED check).
Extrapolation: R(L) = R_inf + B L^-w (+ C L^-2w), w = 2(2-K) from the non-growing O(mu^2) shift of K."""
import csv, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = list(csv.DictReader(open(os.path.join(HERE, "ratio_ed.csv")))); alpha = {1.0: 27.364, 1.5: 17.089}
for K in (1.0, 1.5):
    d = {}
    for r in R:
        if float(r["K"]) == K:
            d.setdefault(int(r["N"]), {})[float(r["mu"])] = r
    Ns = sorted(d); L = np.array(Ns) / np.sqrt(2)
    rat = np.array([float(d[N][1e-3]["delta_pi"]) ** 2 / (float(d[N][1e-3]["x01"]) - float(d[N][0.0]["x01"])) for N in Ns])
    w = 2 * (2 - K) if K != 1.0 else 2.0
    for nmin in (10, 14):
        m = np.array(Ns) >= nmin
        c = np.linalg.lstsq(np.vstack([np.ones(m.sum()), L[m] ** -w, L[m] ** (-2 * w)]).T, rat[m], rcond=None)[0]
        print(f"K={K} N>={nmin}: R_inf={c[0]:.4f}  (8pi^2/alpha={8*np.pi**2/alpha[K]:.4f})")
