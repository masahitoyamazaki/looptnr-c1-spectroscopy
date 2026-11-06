"""Analysis of runs B1 (staggered seed) and B2 (fine chi scan).

Usage:  python3 b1_b2.py FOLLOWUP_OUT_DIR       (B1_*/, B1b_*/ and B2_*/ with levels.csv, steps.csv)
The h = 0 runs and chi = 24, 32, 40, 48 are taken from ../../cpu_baseline_20261106/data.
For each run the O(2) charged-level deviation X = 1 - 4K x_{0,1} is fitted as in analyze.py
(n = 12 to the step before |X| first exceeds 0.1), giving g_d^2(n=12) = -A r^{12-n_e} / (4K alpha_01)
and the end of the weak-coupling window n_e.
B1: with the staggered weight c_v(1 +- h), the dimerization coupling at n = 12 is g_seed + c h, so
    g_d^2(12; h) = (g_seed + c h)^2: a fit over h gives the signed seed relative to the direction of h,
    the calibration c, and the compensating h* = -g_seed / c.
"""
import csv, math, os, sys
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "main", "analysis"))
from analyze import load, level, fit_growth, ALPHA   # noqa: E402

BASE = os.path.join(HERE, "..", "..", "main", "data")
alpha = {round(float(r["K"]), 4): r for r in csv.DictReader(open(ALPHA))}


def seed(d, K):
    lev, _ = load(d)
    X = {n: 1 - 4 * K * level(rows, "(1, 2)", 1) for n, rows in lev.items()}
    out = [m for m in sorted(X) if m >= 12 and abs(X[m]) > 0.1]
    if not out:
        return None
    ne = out[0] - 1
    n = np.arange(12, ne + 1, dtype=float)
    _, lr, c, _ = fit_growth(n, np.array([X[int(m)] for m in n]), ne, False, K)
    g2 = -c[2] * math.exp(lr * (12 - ne)) / (4 * K * float(alpha[round(K, 4)]["m1"]))
    return dict(ne=ne, r=math.exp(lr), g2=g2)


def main(out_dir):
    rows = []
    print("== B1: g_d^2(n=12) and n_e versus the staggered deformation h")
    for K in (1.0, 1.5):
        for chi in (32, 40):
            pts = []
            base = seed(os.path.join(BASE, f"CU1_K{K:.4f}_chi{chi}"), K)
            pts.append((0.0, base))
            # B1: |h| <= 1e-4; B1b (2026-11-06, jobB1b): 3e-4 <= |h| <= 3e-2
            for m in (1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2):
                for sgn in (1, -1):
                    h = sgn * m
                    tag = "B1" if m <= 1e-4 else "B1b"
                    d = os.path.join(out_dir, f"{tag}_K{K}_chi{chi}_h{h:+.0e}")
                    f = seed(d, K) if os.path.isdir(d) else None
                    if f:
                        pts.append((h, f))
            pts.sort(key=lambda p: p[0])
            hs = np.array([p[0] for p in pts]); g2 = np.array([p[1]["g2"] for p in pts])
            ok = g2 > 0
            best = None
            for s0 in (1, -1):
                g0 = s0 * math.sqrt(max(base["g2"], 1e-30))
                cs = math.sqrt(g2[np.argmax(np.abs(hs))]) / np.abs(hs).max()
                for sc in (1, -1):
                    res = least_squares(lambda p: np.log((p[0] + p[1] * hs[ok]) ** 2 + 1e-40) - np.log(g2[ok]),
                                        [g0, sc * cs])
                    if best is None or res.cost < best.cost:
                        best = res
            g0, c = best.x
            if c < 0:            # report with c > 0: the sign of g_seed is relative to the direction of h
                g0, c = -g0, -c
            hstar = -g0 / c
            print(f" K={K} chi={chi}: g_seed(12) = {g0:+.3e} (|g_seed|^2 = {g0**2:.2e}, h=0 run {base['g2']:.2e}), "
                  f"c = {c:.3e} per unit h, h* = {hstar:+.2e}, rms log residual {math.sqrt(2 * best.cost / ok.sum()):.2f}")
            for h, f in pts:
                pred = (g0 + c * h) ** 2
                print(f"    h={h:+.0e}: g2={f['g2']:.2e} (model {pred:.2e}), n_e={f['ne']}, r/2^(2-K)={f['r'] / 2 ** (2 - K):.4f}")
                rows.append(dict(set="B1", K=K, chi=chi, h=h, g2_n12=f["g2"], n_e=f["ne"], r_over_pred=f["r"] / 2 ** (2 - K),
                                 g_seed=g0, c=c, h_star=hstar))
    print("== B2: g_d^2(n=12) and n_e versus chi (every integer from 24 to 48)")
    for K in (1.0, 1.5):
        line = []
        for chi in range(24, 49):
            d = os.path.join(BASE, f"CU1_K{K:.4f}_chi{chi}") if chi in (24, 32, 40, 48) else os.path.join(out_dir, f"B2_K{K}_chi{chi}")
            f = seed(d, K) if os.path.isdir(d) else None
            if f:
                line.append(f"{chi}:{f['g2']:.1e}/{f['ne']}")
                rows.append(dict(set="B2", K=K, chi=chi, h=0.0, g2_n12=f["g2"], n_e=f["ne"], r_over_pred=f["r"] / 2 ** (2 - K)))
        print(f" K={K}: chi:g2/n_e  " + "  ".join(line))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(HERE, "b1_b2.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1])
