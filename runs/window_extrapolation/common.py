"""Shared helpers: locate runs and refit X(n), X_J(n) with the paper's protocol for an arbitrary upper cut.

Reuses load / level / fit_growth from ../main/analysis/analyze.py (unmodified).
The fit is identical to analyze.analyze_o2 / paper_figures.o2_fit: window n_s..n_e with n_e the step
before |X| first exceeds X_max (n >= n_s); columns 1, t, r^(n-n_e) [+ (2^(2-4K))^(n-12) for K < 3/4];
r on the same 4000-point grid in ln r over [ln 1.02, ln 4]; A_J from linear least squares with the same columns.
"""
import csv, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "main")
FOLLOW = os.path.join(HERE, "..", "additional", "data")
sys.path.insert(0, os.path.join(BASE, "analysis"))
from analyze import ALPHA, load, level, fit_growth  # noqa: E402

ALPHAS = {}
for r in csv.DictReader(open(ALPHA)):
    ALPHAS.setdefault(round(float(r["K"]), 4), r)
LNR_GRID = (math.log(1.02), math.log(4))


def run_dir(K, chi):
    if chi == 96:
        return os.path.join(FOLLOW, f"B5_K{K:.1f}_chi96")
    if K < 0.7:
        return os.path.join(FOLLOW, f"B8_K{K:g}_chi{chi}")
    return os.path.join(BASE, "data", f"CU1_K{K:.4f}_chi{chi}")


def series(K, chi):
    lev, steps = load(run_dir(K, chi))
    X, XJ = {}, {}
    for n, rows in lev.items():
        X[n] = 1 - 4 * K * level(rows, "(1, 2)", 1)
        v = sorted(float(r["x"]) for r in rows if "(0, 1)" in r["sector"] and int(r["rank"]) <= 3)
        if len(v) == 3:
            XJ[n] = 1 - min(((abs(v[i] - v[j]), (v[i] + v[j]) / 2) for i in range(3) for j in range(i + 1, 3)))[1]
    return X, XJ, lev, steps


def fit_cut(K, X, XJ, xmax, ns=12):
    """Paper protocol with upper cut xmax. Returns dict or None when no exit from the window exists."""
    out = [m for m in sorted(X) if m >= ns and abs(X[m]) > xmax]
    if not out:
        return None
    ne = out[0] - 1
    n = np.arange(ns, ne + 1, dtype=float)
    npts = len(n)
    umk = K < 0.75
    row = dict(ne=ne, npts=npts, umklapp=int(umk))
    if npts < (6 if umk else 5):
        return row
    _, lr, c, M = fit_growth(n, np.array([X[int(m)] for m in n]), ne, umk, K)
    q = float("nan")
    if round(K, 4) in ALPHAS and all(int(m) in XJ for m in n):
        cj, *_ = np.linalg.lstsq(M, np.array([XJ[int(m)] for m in n]), rcond=None)
        a = ALPHAS[round(K, 4)]
        q = (cj[2] / c[2]) / (float(a["J"]) / (4 * K * float(a["m1"])))
    edge = min(abs(lr - LNR_GRID[0]), abs(lr - LNR_GRID[1])) < 1e-3
    row.update(r=math.exp(lr), rate_ratio=math.exp(lr) / 2 ** (2 - K), amp_ratio=q, A=c[2], r_at_grid_edge=int(edge))
    return row
