"""Extrapolate the exact-diagonalization coupling g_d(L)/mu (rho_ed.csv) to the circumference of the standard
readout cylinder at step 12 of LoopTNR, L_12 = 2 sqrt2 * 2^6 a_0, and compare with rho of Paper 1b, Table III.

g(L)/mu = A (L/2pi)^(2-K) [1 + B (L/2pi)^(-omega)], fitted separately to N = 0 and 2 mod 4, for omega = 1, 2
and for the largest sizes only; mu-linearity is checked with mu = +-1e-4 and 1e-3.
Output: rho_fit.csv
"""
import csv, os, numpy as np
from scipy.optimize import curve_fit
HERE = os.path.dirname(os.path.abspath(__file__))
# LoopTNR rho at chi = 32, 40 (Paper 1b, Table III), read from the global parabola fits of seed_refit
for _d in ("seed_refit_20261112", "seed_refit"):
    _p = os.path.join(HERE, "..", _d, "refit_b9_table.csv")
    if os.path.exists(_p):
        break
_t = {(float(r["K"]), int(r["chi"])): float(r["rho"]) for r in csv.DictReader(open(_p))}
RHO_TNR = {K: (_t[(K, 32)], _t[(K, 40)]) for K in (0.75, 1.0, 1.25, 1.5, 1.75)}  # chi=32, 40
L12 = 2 * np.sqrt(2) * 2 ** 6
rows = list(csv.DictReader(open(os.path.join(HERE, "rho_ed.csv"))))
out = []
for K in sorted({float(r["K"]) for r in rows}):
    R = [r for r in rows if float(r["K"]) == K]
    g = {}
    for r in R:
        g.setdefault(int(r["N"]), {})[float(r["mu"])] = float(r["g"])
    Ns = sorted(g)
    lin = max(abs((g[N][1e-3] / 1e-3) / (g[N][1e-4] / 1e-4) - 1) for N in Ns)
    odd = max(abs(g[N][1e-4] + g[N][-1e-4]) / abs(g[N][1e-4]) for N in Ns)
    y = np.array([abs(g[N][1e-4]) / 1e-4 for N in Ns]); L = np.array(Ns) / np.sqrt(2); s = L / (2 * np.pi)
    est = {}
    for cls in (0, 2):
        m = np.array([N % 4 == cls for N in Ns])
        for om in (1.0, 2.0):
            f = lambda s, A, B: A * s ** (2 - K) * (1 + B * s ** (-om))
            for nmin in (8, 12):
                mm = m & (np.array(Ns) >= nmin)
                if mm.sum() < 3:
                    continue
                p, _ = curve_fit(f, s[mm], y[mm], p0=(1, 0))
                est[(cls, om, nmin)] = p[0] * (L12 / (2 * np.pi)) ** (2 - K)
    v = np.array(list(est.values()))
    naive = y[-1] * (L12 / L[-1]) ** (2 - K)
    kt = RHO_TNR[K]
    out.append(dict(K=K, rho_ed=np.median(v), rho_ed_min=v.min(), rho_ed_max=v.max(), rho_ed_largestN=naive,
                    rho_loop_chi32=kt[0], rho_loop_chi40=kt[1], ratio=np.mean(kt) / np.median(v),
                    mu_nonlinearity=lin, mu_oddness_dev=odd))
    print(f"K={K}: rho_ED={np.median(v):.3f} [{v.min():.3f},{v.max():.3f}] (largest N only {naive:.3f}); "
          f"LoopTNR {kt}; ratio {np.mean(kt)/np.median(v):.3f}; nonlin(1e-3) {lin:.1e}; odd {odd:.1e}")
    for N, yy in zip(Ns, y):
        print(f"   N={N:2d} L={N/np.sqrt(2):6.2f} g/mu={yy:.5f}  g/mu*(L12/L)^(2-K)={yy*(L12/(N/np.sqrt(2)))**(2-K):.3f}")
with open(os.path.join(HERE, "rho_fit.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
