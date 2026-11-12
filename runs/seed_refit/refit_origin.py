"""Seeds of the branched (B3), loop-optimization (B4), and single-step-replacement (C7) runs read with the protocol
of paper 1b: fit X = c0 + c1 n + A r^(n - n_end) with r = 2^(2-K) fixed over the contiguous steps n >= 12 with
|X| <= 0.02, g_d^2(12) = -A r^(12 - n_end) / (4 K alpha_01).  Ratios to the reference runs (main).
Also: wall-clock RG time to the end of the window, from steps.csv (rg_seconds), for the cost comparison.
Outputs: refit_origin.csv, cost.csv
"""
import csv, gzip, math, os, sys
import numpy as np
RUNS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(RUNS, "main", "data")
FB = os.path.join(RUNS, "additional", "data")
ALPHA = {round(float(r["K"]), 4): float(r["m1"]) for r in csv.DictReader(open(os.path.join(RUNS, "main", "analysis", "sixvertex_g2_alpha.csv")))}

def rd(d, name):
    p = os.path.join(d, name)
    return csv.DictReader(open(p) if os.path.exists(p) else gzip.open(p + ".gz", "rt"))

def X_of(d, K):
    X = {}
    for r in rd(d, "levels.csv"):
        if "(1, 2)" in r["sector"] and int(r["rank"]) == 1:
            X[int(r["step"])] = 1 - 4 * K * float(r["x"])
    return X

def seed(d, K, cut=0.02):
    X = X_of(d, K); r = 2 ** (2 - K)
    n = 12; win = []
    while n in X and abs(X[n]) <= cut:
        win.append(n); n += 1
    if len(win) < 4:
        return None
    ne_ = win[-1]; nn = np.array(win, float); x = np.array([X[m] for m in win])
    M = np.vstack([np.ones_like(nn), nn, r ** (nn - ne_)]).T
    c, *_ = np.linalg.lstsq(M, x, rcond=None)
    g2 = -c[2] * r ** (12 - ne_) / (4 * K * ALPHA[round(K, 4)])
    out = [m for m in sorted(X) if m >= 12 and abs(X[m]) > 0.1]
    return dict(g2=g2, n_end=ne_, n_e=(out[0] - 1) if out else None)

def rgtime(d, upto):
    t = 0.0
    for r in rd(d, "steps.csv"):
        if int(r["step"]) <= upto:
            t += float(r["rg_seconds"])
    return t

rows = []
for K in (1.0, 1.5):
    ref = seed(os.path.join(BASE, f"CU1_K{K:.4f}_chi40"), K)
    for n0 in (6, 10):
        for var in ("same", "chi32", "chi48", "s80c9", "ef12"):
            d = os.path.join(FB, f"B3_K{K}_n{n0}_{var}")
            if os.path.isdir(d):
                f = seed(d, K); rows.append(dict(set="B3", K=K, chi=40, label=f"n0={n0} {var}", ratio=f["g2"] / ref["g2"], g2=f["g2"], n_e=f["n_e"]))
for K, chi in ((1.5, 24), (1.5, 40), (1.0, 40)):
    ref = seed(os.path.join(BASE, f"CU1_K{K:.4f}_chi{chi}"), K)
    for var in ("s80c12", "s320c12"):
        d = os.path.join(FB, f"B4_K{K}_chi{chi}_{var}")
        if os.path.isdir(d):
            f = seed(d, K); rows.append(dict(set="B4", K=K, chi=chi, label=var, ratio=f["g2"] / ref["g2"], g2=f["g2"], n_e=f["n_e"]))
ref = seed(os.path.join(BASE, "CU1_K1.5000_chi40"), 1.5)
for n in range(5, 12):
    for v in ("chi64", "s80", "s0"):
        d = os.path.join(FB, f"C7_K1.5_n{n}_{v}")
        if os.path.isdir(d):
            f = seed(d, 1.5)
            rows.append(dict(set="C7", K=1.5, chi=40, label=f"n={n} {v}", ratio=(f["g2"] / ref["g2"]) if f else float("nan"), g2=f["g2"] if f else float("nan"), n_e=f["n_e"] if f else None))
w = csv.DictWriter(open("refit_origin.csv", "w"), fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
for r in rows:
    print(f"{r['set']} K={r['K']} chi={r['chi']} {r['label']:14s} ratio={r['ratio']:.2f} n_e={r['n_e']}")

# ---- cost: wall-clock RG seconds to reach the end of the weak-coupling window
print("\n== cost (sum of rg_seconds up to n_e)")
cost = []
def add(tag, d, K):
    f = seed(d, K)
    cost.append(dict(tag=tag, K=K, n_e=f["n_e"], rg_seconds=rgtime(d, f["n_e"]), per_step=rgtime(d, f["n_e"]) / f["n_e"]))
for K in (1.0, 1.5):
    for chi in (24, 32, 40, 48, 64, 96):
        d = os.path.join(BASE, f"CU1_K{K:.4f}_chi{chi}")
        if not os.path.isdir(d):
            d = os.path.join(FB, f"B5_K{K}_chi{chi}")
        if os.path.isdir(d):
            add(f"chi{chi} uncompensated", d, K)
for K, chi, mu in ((1.5, 32, "3.5e-05"), (1.5, 40, "2e-05"), (1.0, 40, "4e-06"), (1.5, 24, "-0.0001")):
    for cand in (f"B9_K{K}_chi{chi}_ab+{mu}", f"B9_K{K}_chi{chi}_ab{mu}"):
        d = os.path.join(FB, cand)
        if os.path.isdir(d):
            add(f"chi{chi} compensated mu={mu}", d, K); break
w = csv.DictWriter(open("cost.csv", "w"), fieldnames=list(cost[0])); w.writeheader(); [w.writerow(c) for c in cost]
for c in cost:
    print(f"K={c['K']} {c['tag']:30s} n_e={c['n_e']}  RG time to n_e = {c['rg_seconds']/3600:.2f} h  ({c['per_step']:.0f} s/step)")
