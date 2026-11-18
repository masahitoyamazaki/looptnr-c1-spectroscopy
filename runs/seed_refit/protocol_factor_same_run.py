"""Protocol factor of the seed on the same runs: g_d^2(12) read from the main O(2) spectra with the protocol of Paper II
(rate fixed to 2^(2-K), window from n=12 while |X|<=0.02, background beta0+beta1 n) divided by the seed of the main fits
of Paper I (o2_fits.csv, column gd2_n12) of the same run.  Paper I, App. C: 0.7-1.6 for K<=1.5 and 4-5 at K=1.75;
at fixed K it varies by less than 8% between chi=32 and 40 and by up to 30% when chi=24 is included.
Output: protocol_factor_same_run.csv
"""
import csv, gzip, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "..", "main")  # public repository layout
if not os.path.isdir(MAIN):  # collaboration repository layout
    MAIN = os.path.join(HERE, "..", "main")
ALPHA = {round(float(r["K"]), 4): float(r["m1"]) for r in csv.DictReader(open(os.path.join(MAIN, "analysis", "sixvertex_g2_alpha.csv")))}
FIT = {(round(float(r["K"]), 3), int(r["chi"])): float(r["gd2_n12"])
       for r in csv.DictReader(open(os.path.join(MAIN, "analysis", "o2_fits.csv"))) if r["gd2_n12"] not in ("", "nan")}

def levels(d):
    p = os.path.join(d, "levels.csv")
    return csv.DictReader(open(p) if os.path.exists(p) else gzip.open(p + ".gz", "rt"))

def seed(d, K, cut=0.02):
    X = {int(r["step"]): 1 - 4 * K * float(r["x"]) for r in levels(d) if "(1, 2)" in r["sector"] and int(r["rank"]) == 1}
    r, n, win = 2 ** (2 - K), 12, []
    while n in X and abs(X[n]) <= cut:
        win.append(n); n += 1
    if len(win) < 4:
        return None
    ne, nn = win[-1], np.array(win, float)
    c, *_ = np.linalg.lstsq(np.vstack([np.ones_like(nn), nn, r ** (nn - ne)]).T, np.array([X[m] for m in win]), rcond=None)
    return -c[2] * r ** (12 - ne) / (4 * K * ALPHA[round(K, 4)])

rows = []
for K in (0.75, 1.0, 1.25, 1.5, 1.75):
    for chi in (24, 32, 40, 48, 64):
        g = seed(os.path.join(MAIN, "data", f"CU1_K{K:.4f}_chi{chi}"), K)
        if g is not None and (round(K, 3), chi) in FIT:
            rows.append({"K": K, "chi": chi, "g2_12_fixed": g, "g2_12_main": FIT[(round(K, 3), chi)], "factor": g / FIT[(round(K, 3), chi)]})
with open(os.path.join(HERE, "protocol_factor_same_run.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for K in sorted({d["K"] for d in rows}):
    fs = {d["chi"]: d["factor"] for d in rows if d["K"] == K}
    a = [fs[c] for c in (24, 32, 40) if c in fs]; b = [fs[c] for c in (32, 40) if c in fs]
    print(f"K={K:<5}", " ".join(f"chi={c}:{v:.3f}" for c, v in fs.items()),
          f"| chi=24-40 {100*(max(a)-min(a))/min(a):.1f}% | chi=32-40 {100*(max(b)-min(b))/min(b):.1f}%")
