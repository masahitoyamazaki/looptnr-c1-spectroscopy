"""T1: tube-width check (2026-11-06).  The saved O(2) checkpoints at chi = 40 (K = 1, 1.5) are re-read with the wider
tube [4/sqrt(10), 2 sqrt(10), 2/sqrt(10)] (circumference sqrt(5) times the standard one; ../wide_readout.jl) and
analyzed with the same fit as the standard readout (analyze_o2 of ../../cpu_baseline_20261106/analysis/analyze.py).
Expected if the deviations come from a single dimerization coupling: the same rate r and the same normalized amplitude
ratio, and a coupling g_d^2(n) larger by (sqrt 5)^{2(2-K)} = 5^{2-K} at the same step n.
Usage:  python3 wide_tube.py
"""
import csv, gzip, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "main", "analysis"))
from analyze import fit_growth, analyze_o2, ALPHA   # noqa: E402

DATA = os.path.join(HERE, "..", "data")
BASE = os.path.join(HERE, "..", "..", "main", "data")


def wide_levels(K, chi):
    lev = {}
    with gzip.open(os.path.join(DATA, f"W1_K{K:.4f}_chi{chi}", "levels_wide.csv.gz"), "rt") as f:
        for r in csv.DictReader(f):
            lev.setdefault(int(r["step"]), []).append(r)
    return lev


def fit_wide(K, chi, alpha):
    lev = wide_levels(K, chi)
    def x01(rows):
        return min(float(r["x"]) for r in rows if "(1, 2)" in r["sector"])
    def xJ(rows):  # the two lowest spin-one levels of the neutral C-odd sector: the currents
        v = sorted(float(r["x"]) for r in rows if "(0, 1)" in r["sector"] and abs(abs(float(r["spin"])) - 1) < 0.3)
        return sum(v[:2]) / 2
    X = {n: 1 - 4 * K * x01(rows) for n, rows in lev.items()}
    out = [m for m in sorted(X) if m >= 12 and abs(X[m]) > 0.1]
    n_e = out[0] - 1
    n = np.arange(12, n_e + 1, dtype=float)
    x = np.array([X[int(m)] for m in n])
    _, lr, c, M = fit_growth(n, x, n_e, K < 0.75, K)
    xj = np.array([1 - xJ(lev[int(m)]) for m in n])
    cj, *_ = np.linalg.lstsq(M, xj, rcond=None)
    a01, aJ = float(alpha["m1"]), float(alpha["J"])
    return dict(n_e=n_e, r=math.exp(lr), r_over_pred=math.exp(lr) / 2 ** (2 - K),
                ratio_over_pred=(cj[2] / c[2]) / (aJ / (4 * K * a01)), A=c[2],
                gd2_n12=-c[2] * math.exp(lr * (12 - n_e)) / (4 * K * a01))


def main():
    import glob
    alpha = {round(float(r["K"]), 4): r for r in csv.DictReader(open(ALPHA))}
    rows = []
    for d in sorted(glob.glob(os.path.join(DATA, "W1_K*_chi*"))):
        K = float(d.split("_K")[1].split("_")[0]); chi = int(d.split("_chi")[1])
        a = alpha[round(K, 4)]
        nar = analyze_o2(os.path.join(BASE, f"CU1_K{K:.4f}_chi{chi}"), K, chi, a)
        wid = fit_wide(K, chi, a)
        nc = min(nar["n_e"], wid["n_e"])   # latest step inside both fit windows: no long extrapolation
        g_n = -nar["A"] * nar["r"] ** (nc - nar["n_e"]); g_w = -wid["A"] * wid["r"] ** (nc - wid["n_e"])
        row = dict(K=K, chi=chi, ne_std=nar["n_e"], ne_wide=wid["n_e"], r_std=nar["r_over_pred"], r_wide=wid["r_over_pred"],
                   q_std=nar["ratio_over_pred"], q_wide=wid["ratio_over_pred"], g2_ratio_nc=g_w / g_n, pred=5 ** (2 - K))
        rows.append(row)
        print(f"K={K:<5} chi={chi:<3} n_e {row['ne_std']}/{row['ne_wide']}  r/2^(2-K) {row['r_std']:.4f}/{row['r_wide']:.4f}  "
              f"ratio {row['q_std']:.4f}/{row['q_wide']:.4f}  g2 wide/std at n={nc}: {row['g2_ratio_nc']:.3f} (pred {row['pred']:.3f})")
    with open(os.path.join(HERE, "wide_tube.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
