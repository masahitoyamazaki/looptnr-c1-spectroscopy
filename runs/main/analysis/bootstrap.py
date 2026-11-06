"""Residual (moving-block) bootstrap for the O(2) growth factor r and the normalized amplitude ratio (2026-11-06).

Same fit as analyze_o2 in analyze.py (window n = 12 .. n_e, linear background, growth term with free rate r,
umklapp term for K < 0.75; the current level X_J fitted with the same rate and background).  The residuals of the
two series are resampled in pairs (same steps) as moving blocks of BLOCK consecutive steps (BLOCK=1: i.i.d.), r is refitted on a fine grid around the best value, and the
amplitude ratio is recomputed.  Residuals are correlated between steps, so these are statistical errors in the
narrow sense; the window variations quoted in the paper remain the dominant (systematic) uncertainty.
Usage:  python3 bootstrap.py [NBOOT] [BLOCK]      (reads ../data/CU1_K*_chi*; default BLOCK=4)
"""
import csv, glob, math, os, sys
import numpy as np
from analyze import load, level, fit_growth, ALPHA

HERE = os.path.dirname(os.path.abspath(__file__))


def series(d, K):
    lev, _ = load(d)
    X = {n: 1 - 4 * K * level(rows, "(1, 2)", 1) for n, rows in lev.items()}
    out = [m for m in sorted(X) if m >= 12 and abs(X[m]) > 0.1]
    if not out:
        return None
    n_e = out[0] - 1
    n = np.arange(12, n_e + 1, dtype=float)
    x = np.array([X[int(m)] for m in n])
    xj = []
    for m in n:
        v = sorted(float(r["x"]) for r in lev[int(m)] if "(0, 1)" in r["sector"] and int(r["rank"]) <= 3)
        pairs = [(abs(v[i] - v[j]), (v[i] + v[j]) / 2) for i in range(3) for j in range(i + 1, 3)]
        xj.append(1 - min(pairs)[1])
    return n, x, np.array(xj), n_e


def design(n, n_e, lr, K):
    t = (n - (12 + n_e) / 2) / 10
    cols = [np.ones_like(t), t, np.exp(lr * (n - n_e))]
    if K < 0.75:
        cols.append((2.0 ** (2 - 4 * K)) ** (n - 12))
    return np.vstack(cols).T


def refit(n, x, n_e, K, lr0):
    best = None
    for lr in np.linspace(lr0 - 0.15, lr0 + 0.15, 1200):
        M = design(n, n_e, lr, K)
        c, *_ = np.linalg.lstsq(M, x, rcond=None)
        rss = ((M @ c - x) ** 2).sum()
        if best is None or rss < best[0]:
            best = (rss, lr, c, M)
    return best


def main(nboot, block):
    alpha = {round(float(r["K"]), 4): r for r in csv.DictReader(open(ALPHA))}
    rng = np.random.default_rng(20261106)
    rows = []
    for d in sorted(glob.glob(os.path.join(HERE, "..", "data", "CU1_K*_chi*"))):
        name = os.path.basename(d)
        K = float(name.split("_K")[1].split("_")[0]); chi = int(name.split("chi")[1])
        if K >= 2 or chi < 24:
            continue
        s = series(d, K)
        if s is None:
            continue
        n, x, xj, n_e = s
        a = alpha[round(K, 4)]; pred = float(a["J"]) / (4 * K * float(a["m1"]))
        _, lr, c, M = fit_growth(n, x, n_e, K < 0.75, K)
        cj, *_ = np.linalg.lstsq(M, xj, rcond=None)
        res, resj = x - M @ c, xj - M @ cj
        rs, qs = [], []
        for _ in range(nboot):
            starts = rng.integers(0, len(n) - block + 1, (len(n) + block - 1) // block)
            idx = np.concatenate([np.arange(st, st + block) for st in starts])[:len(n)]
            xb, xjb = M @ c + res[idx], M @ cj + resj[idx]
            _, lrb, cb, Mb = refit(n, xb, n_e, K, lr)
            cjb, *_ = np.linalg.lstsq(Mb, xjb, rcond=None)
            rs.append(math.exp(lrb) / 2 ** (2 - K)); qs.append((cjb[2] / cb[2]) / pred)
        row = dict(K=K, chi=chi, n_e=n_e, r_over_pred=math.exp(lr) / 2 ** (2 - K), sd_r=float(np.std(rs, ddof=1)),
                   ratio_over_pred=(cj[2] / c[2]) / pred, sd_ratio=float(np.std(qs, ddof=1)))
        rows.append(row)
        print(f"K={K:<5} chi={chi:<3} n_e={n_e:<3} r/2^(2-K)={row['r_over_pred']:.4f} +- {row['sd_r']:.4f}   "
              f"ratio={row['ratio_over_pred']:.4f} +- {row['sd_ratio']:.4f}", flush=True)
    with open(os.path.join(HERE, f"bootstrap_block{block}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400, int(sys.argv[2]) if len(sys.argv) > 2 else 4)
