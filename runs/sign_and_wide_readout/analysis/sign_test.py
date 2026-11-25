"""Sign of the truncation seed at chi=48, 64 (followup C, 2026-11-10).

Runs S_K{K}_chi{chi}_ab{p|m}{mu} (STAGGER_AB=+-mu, O(2), K=1, 1.5) together with the mu=0 runs of
main.  g_d^2(12) and n_e as in followup_B analysis/b1_b2.seed; fit (g_seed + rho mu)^2 (rho: response, called kappa in early drafts)
with rho > 0, so sign(g_seed) is the sign of the seed relative to the direction of mu, comparable with
followup_B analysis/b9.csv (chi = 24, 32, 40).
Usage: python3 sign_test.py   -> prints the table, writes sign_test.csv
"""
import csv, glob, math, os, sys
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(RUNS, "additional", "analysis"))
from b1_b2 import seed  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
BASE = os.path.join(RUNS, "main", "data")


def fit(hs, g2, g0guess):
    best = None
    for s0 in (1, -1):
        for sc in (1, -1):
            p0 = [s0 * math.sqrt(g0guess), sc * math.sqrt(g2.max()) / abs(hs).max()]
            r = least_squares(lambda p: np.log((p[0] + p[1] * hs) ** 2 + 1e-40) - np.log(g2), p0)
            if best is None or r.cost < best.cost:
                best = r
    g0, k = best.x
    if k < 0:
        g0, k = -g0, -k
    return g0, k, math.sqrt(2 * best.cost / len(hs))


rows = []
for K in (1.0, 1.5):
    for chi in (48, 64):
        pts = [(0.0, seed(os.path.join(BASE, f"CU1_K{K:.4f}_chi{chi}"), K))]
        for d in glob.glob(os.path.join(DATA, f"S_K{K}_chi{chi}_ab*")):
            t = os.path.basename(d).split("_ab")[1]
            mu = (1 if t[0] == "p" else -1) * float(t[1:])
            pts.append((mu, seed(d, K)))
        pts.sort(key=lambda p: p[0])
        ok = [(h, f) for h, f in pts if f]
        hs = np.array([h for h, _ in ok]); g2 = np.array([f["g2"] for _, f in ok])
        g0, k, rms = fit(hs, g2, pts[0][1]["g2"] if pts[0][0] == 0 else g2.min())
        g0, k, rms = fit(hs, g2, [f["g2"] for h, f in ok if h == 0][0])
        print(f"K={K} chi={chi}: g_seed={g0:+.3e} rho={k:.3f} mu*={-g0/k:+.2e} rms(log)={rms:.2f}")
        for h, f in ok:
            print(f"   mu={h:+.1e} g2={f['g2']:.3e} model={(g0+k*h)**2:.3e} n_e={f['ne']} r/2^(2-K)={f['r']/2**(2-K):.4f}")
            rows.append(dict(K=K, chi=chi, mu=h, g2_n12=f["g2"], n_e=f["ne"], r_over_pred=f["r"] / 2 ** (2 - K),
                             g_seed=g0, rho=k, mu_star=-g0 / k, rms_log=rms))
with open(os.path.join(HERE, "sign_test.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
