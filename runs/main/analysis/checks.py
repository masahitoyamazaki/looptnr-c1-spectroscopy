"""Additional analyses for main_en.tex (single-environment data):
 (1) U(1) charge-averaged part E_1: growth rate and G = g_d^2 + g_s^2;
 (2) U(1) charge proportionality O_q = q B_n for q = 1, 2, 3;
 (3) O(2) growth rate versus the upper cut of the weak-coupling window;
 (4) K = 2.5: which level carries the 2e-3 offset;
 (6) closed-form alpha coefficients versus the table.
Usage: python3 checks.py   (writes checks.txt)"""
import csv, math, os, sys
import numpy as np
from scipy.special import gamma as G
from analyze import ALPHA, level, fit_growth
from paper_figures import run, u1_window, o2_series, KS, CHIS, alpha

out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

p("(1) U(1) charge-averaged part: X_E = 1 - 4K E_1 fitted from n = 12 to the end of the U(1) window")
for K in KS:
    for chi in (32, 40, 48, 64):
        lev, good = u1_window(K, chi)
        if not good:
            continue
        ne = good[-1][0]
        XE = {}
        for m in range(12, ne + 1):
            xp, xm = level(lev[m], "U₁](1)", 1), level(lev[m], "U₁](-1)", 1)
            XE[m] = 1 - 4 * K * (xp + xm) / 2
        n = np.arange(12, ne + 1, dtype=float)
        rss, lr, c, M = fit_growth(n, np.array([XE[int(m)] for m in n]), ne, False, K)
        Gd = -c[2] / (4 * K * float(alpha[round(K, 4)]["m1"])) if K < 2 else float("nan")
        resid = math.sqrt(rss / len(n))
        p(f"  K={K} chi={chi}: window 12-{ne}, r/2^(2-K)={math.exp(lr) / 2 ** (2 - K):.3f}, A={c[2]:+.2e}, "
          f"|A|/rms={abs(c[2]) / resid:.0f}, G(n_e)={Gd:.2e}")

p("(2) U(1) charge proportionality over the fit window (last 10 points)")
for K in KS:
    for chi in (40, 64):
        lev, good = u1_window(K, chi)
        if len(good) < 10:
            continue
        num = den = 0.0
        for m, *_ in good[-10:]:
            Oq = [(level(lev[m], f"U₁]({q})", 1) - level(lev[m], f"U₁](-{q})", 1)) / 2 for q in (1, 2, 3)]
            q = np.array([1, 2, 3.0]); B = (q @ np.array(Oq)) / (q @ q)
            num += ((np.array(Oq) - q * B) ** 2).sum(); den += (np.array(Oq) ** 2).sum()
        p(f"  K={K} chi={chi}: eps_ch = {100 * math.sqrt(num / den):.2f}%")

p("(3) O(2) growth rate r/2^(2-K) versus the upper cut |X| <= cut (n_s = 12)")
for K in (0.75, 1.0, 1.25, 1.5):
    for chi in (40, 64):
        X, _ = o2_series(K, chi); row = []
        for cut in (0.01, 0.02, 0.05, 0.10):
            big = [m for m in sorted(X) if m >= 12 and abs(X[m]) > cut]
            ne = big[0] - 1
            n = np.arange(12, ne + 1, dtype=float)
            _, lr, c, _ = fit_growth(n, np.array([X[int(m)] for m in n]), ne, False, K)
            row.append(f"cut {cut}: {math.exp(lr) / 2 ** (2 - K):.4f} (n_e={ne})")
        p(f"  K={K} chi={chi}: " + "; ".join(row))

p("(4) K = 2.5, chi = 40, 64: relative deviations averaged over n = 20-80")
for chi in (40, 64):
    lev, _ = run("CU1", 2.5, chi)
    rows = []
    for m in range(20, 81):
        if m not in lev: continue
        r = lev[m]
        x01 = level(r, "(1, 2)", 1); x02 = level(r, "(2, 2)", 1)
        odd = sorted(float(x["x"]) for x in r if "(0, 1)" in x["sector"])
        even = sorted(float(x["x"]) for x in r if "(0, 0)" in x["sector"])[1:]
        xJ = min(odd, key=lambda v: abs(v - 1)); xeO = min(odd, key=lambda v: abs(v - 2.5)); xeE = min(even, key=lambda v: abs(v - 2.5))
        rows.append((4 * 2.5 * x01 - 1, 2.5 * x02 - 1, xJ - 1, (xeE + xeO) / 5 - 1, xeE / 2.5 - 1, xeO / 2.5 - 1))
    a = np.mean(rows, axis=0)
    p(f"  chi={chi}: x01 {a[0]:+.1e}, x02 {a[1]:+.1e}, xJ {a[2]:+.1e}, x_e(avg) {a[3]:+.1e} [E {a[4]:+.1e}, O {a[5]:+.1e}]")

p("(6) alpha coefficients from the closed forms versus the table")
gm = lambda z: G(z) / G(1 - z)
def alpha_em(e, m, K):
    I = lambda e, m: math.pi * G(1 - K) / G(K) * G(K / 2 + e * K + m / 2) * G(K / 2 - e * K + m / 2) / (G(1 - K / 2 + e * K + m / 2) * G(1 - K / 2 - e * K + m / 2))
    return -math.pi * (I(e, m) - I(0, 0))
def alpha_e1(K, sgn):
    return -math.pi ** 2 * (gm(1.5 * K) * gm(1 - K) * gm(-K / 2) + sgn * 0.5 * gm(-K / 2) ** 2 * gm(1 + K) - gm(K / 2) ** 2 * gm(1 - K))
for K in (0.6, 0.75, 1.25, 1.5, 1.75):
    a = alpha[round(K, 4)]
    p(f"  K={K}: alpha_01 {alpha_em(0, 1, K):.4f} (table {float(a['m1']):.4f}); alpha_E {alpha_e1(K, +1):.4f} ({float(a['e1E']):.4f}); alpha_O {alpha_e1(K, -1):.4f} ({float(a['e1O']):.4f})")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "checks.txt"), "w").write("\n".join(out) + "\n")
