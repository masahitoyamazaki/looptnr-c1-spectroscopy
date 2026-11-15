"""Integer chi scan at K = 1.5 with 80 loop-optimization sweeps (cost tolerance 1e-12) against the default scan
(20 sweeps, 1e-9; runs CU1_K1.5000_chiXX of main and B2_K1.5_chiXX of additional).
Seed g_d^2(12) with the protocol of paper 1b (rate 2^(2-K) fixed, |X| <= 0.02 from n = 12), breakdown n_e
(last step with |X| <= 0.1), and xi = L/(2 pi x_{0,1}) on the gapped plateau, L = 2 sqrt2 (sqrt2)^n:
average of the last two steps before the plateau ends (|q - 1| > 0.02 with q = xi_n / xi_{n-1}, or x_{0,1} > 3).
Output: sweep80.csv"""
import csv, gzip, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(RUNS, "seed_refit_20261112"))
K = 1.5
ALPHA = {round(float(r["K"]), 4): float(r["m1"]) for r in csv.DictReader(open(os.path.join(RUNS, "main", "analysis", "sixvertex_g2_alpha.csv")))}

def rd(d):
    p = os.path.join(d, "levels.csv")
    return csv.DictReader(open(p) if os.path.exists(p) else gzip.open(p + ".gz", "rt"))

def x01(d):
    return {int(r["step"]): float(r["x"]) for r in rd(d) if "(1, 2)" in r["sector"] and int(r["rank"]) == 1}

def seed(x):
    X = {n: 1 - 4 * K * v for n, v in x.items()}; r = 2 ** (2 - K); n = 12; win = []
    while n in X and abs(X[n]) <= 0.02:
        win.append(n); n += 1
    ne_ = win[-1]; nn = np.array(win, float); M = np.vstack([np.ones_like(nn), nn, r ** (nn - ne_)]).T
    c, *_ = np.linalg.lstsq(M, np.array([X[m] for m in win]), rcond=None)
    g2 = -c[2] * r ** (12 - ne_) / (4 * K * ALPHA[K])
    out = [m for m in sorted(X) if m >= 12 and abs(X[m]) > 0.1]
    return g2, (out[0] - 1 if out else None)

def xi(x):
    ns = sorted(n for n in x if n >= 12); L = lambda n: 2 * math.sqrt(2) * math.sqrt(2) ** n
    xs = {n: L(n) / (2 * math.pi * x[n]) for n in ns}
    start = next((n for n in ns if abs(1 - 4 * K * x[n]) > 0.5), None)   # well inside the gapped phase
    if start is None: return float("nan")
    prev = None; plateau = []
    for n in [m for m in ns if m >= start]:
        if prev is not None and (abs(xs[n] / xs[prev] - 1) > 0.02 or x[n] > 3):
            if len(plateau) >= 2: break
            plateau = []
        else:
            plateau.append(n)
        prev = n
    return float(np.mean([xs[n] for n in plateau[-2:]])) if len(plateau) >= 2 else float("nan")

rows = []
for chi in range(24, 49):
    d80 = os.path.join(HERE, "out", f"K1.5_chi{chi}_s80c12")
    dd = os.path.join(RUNS, "main", "data", f"CU1_K1.5000_chi{chi}")
    if not os.path.isdir(dd): dd = os.path.join(RUNS, "additional", "data", f"B2_K1.5_chi{chi}")
    a, b = x01(dd), x01(d80)
    (g20, ne20), (g80, ne80) = seed(a), seed(b)
    rows.append(dict(chi=chi, g2_20=g20, g2_80=g80, ratio=g80 / g20, n_e_20=ne20, n_e_80=ne80, xi_20=xi(a), xi_80=xi(b)))
with open(os.path.join(HERE, "sweep80.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print("chi=%2d  g2(12): 20sw %.2e  80sw %.2e  ratio %.2f   n_e %s -> %s   xi %.2e -> %.2e" % (
        r["chi"], r["g2_20"], r["g2_80"], r["ratio"], r["n_e_20"], r["n_e_80"], r["xi_20"], r["xi_80"]))
