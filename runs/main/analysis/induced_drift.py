"""Paper I, App. D ("Induced drift of K"): shift of K_n induced by the dimerization inside the fit window.

From the two levels of Eq. (Kn), x_q = x_{0,1} and x_e = (x_E + x_O)/2 (reconstructed from K_n and eta of
drift_steps.csv: x_q = (1+eta)/(4K_n), x_e = (1+eta)K_n), we subtract the regular parts (alpha_i - alpha_i^pole) g_d^2(n)
of their second-order shifts, with alpha_i^pole = -pi^2 K^2 d_K x_i/(2-K) and g_d^2(n) = -A r^(n-n_e)/(4K alpha_01)
from the growth fit (o2_fits.csv), recompute K_n, and subtract the direct drift (linear fit in the floor window,
drift_summary.csv).  The result at n_e is compared with the fixed-K shift -pi^2 K^2 g_d^2(n_e)/(2-K) of Eq. (flow).
Usage: python3 induced_drift.py   (after analyze.py and drift.py)
"""
import csv, math, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda f: os.path.join(HERE, f)
fits = {(float(r["K"]), int(r["chi"])): r for r in csv.DictReader(open(P("o2_fits.csv")))}
summ = {(float(r["K"]), int(r["chi"])): r for r in csv.DictReader(open(P("drift_summary.csv")))}
alpha = {round(float(r["K"]), 4): r for r in csv.DictReader(open(P("sixvertex_g2_alpha.csv")))}
steps = {}
for r in csv.DictReader(open(P("drift_steps.csv"))):
    steps.setdefault((float(r["K"]), int(r["chi"])), []).append(r)
print("  K  chi  n_e  induced dK_n   fixed-K shift   fraction")
for K in (1.25, 1.5):
    for chi in (40, 64):
        f, s, a = fits[(K, chi)], summ[(K, chi)], alpha[round(K, 4)]
        A, r, ne = float(f["A"]), float(f["r"]), int(f["n_e"])
        a01, ae = float(a["m1"]), (float(a["e1E"]) + float(a["e1O"])) / 2
        p01, pe = math.pi ** 2 / (4 * (2 - K)), -math.pi ** 2 * K ** 2 / (2 - K)
        # direct drift: the linear fit of drift.py, refitted here to get its intercept
        fl = [(int(x["n"]), float(x["K_n"])) for x in steps[(K, chi)]
              if int(s["n_first"]) <= int(x["n"]) <= int(s["n_last"])]
        slope, icpt = np.polyfit([n for n, _ in fl], [k for _, k in fl], 1)
        x = next(x for x in steps[(K, chi)] if int(x["n"]) == ne)
        Kn, sc = float(x["K_n"]), 1 + float(x["eta"])
        xq, xe = sc / (4 * Kn), sc * Kn
        g2 = -A / (4 * K * a01)
        xq, xe = xq - (a01 - p01) * g2, xe - (ae - pe) * g2
        ind = 0.5 * math.sqrt(xe / xq) - (icpt + slope * ne)
        ref = -math.pi ** 2 * K ** 2 * g2 / (2 - K)
        print("%4.2f %4d %4d   %+.4f        %+.4f        %.2f" % (K, chi, ne, ind, ref, ind / ref))
