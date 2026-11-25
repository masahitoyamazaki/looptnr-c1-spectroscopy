"""Referee round 4, major 2: does the drift of K explain the window-length dependence of xi_pred/xi at K=1.75?
For each K=1.75 deformation run (followup_B data), solve per step for K_n and g^2 from
  x_{0,1} = 1/(4K_n) + a_m g^2,   x_e(odd) = K_n + a_O g^2     (second-order coefficients at K=1.75)
fit K_n linearly over the fit window (n=12..n_end), and re-evaluate Eq. (12) and the conversion 4K alpha_{0,1}(K)
of Eq. (8) with K_eff = K_n(n_end) instead of the input K.  Output: kdrift.csv"""
import csv, gzip, math, os, sys, numpy as np
T = sys.argv[1]
D = os.path.join(T, "Otsuki/sixvertex/data"); B = os.path.join(T, "Yamazaki/runs/additional/data")
al = {round(float(r["K"]), 3): r for r in csv.DictReader(open(os.path.join(D, "sixvertex_g2_alpha.csv")))}
Ks = sorted(al)
def alpha(K, key):
    return float(np.interp(K, Ks, [float(al[k][key]) for k in Ks]))
def kappa_sg(K):
    p = K / (2 - K)
    return (math.gamma(p/(p+1)) / math.gamma(1/(p+1)) / math.pi
            * (math.sqrt(math.pi)*math.gamma((p+1)/2) / (2*math.gamma(p/2))) ** (2/(p+1)))
def xi_pred(K, n, g2):
    L = 2*math.sqrt(2)*math.sqrt(2)**n
    return L/(2*math.pi) * (math.sqrt(2)*kappa_sg(K)/math.sqrt(g2)) ** (1/(2-K))
K0 = 1.75; aM, aO = alpha(K0, "m1"), alpha(K0, "e1O")
rows = []
for r in csv.DictReader(open(os.path.join(D, "sixvertex_sg_xi_predict.csv"))):
    if not (r["K"] == "1.75" and r["cut"] == "0.02" and r["rate"] == "fixed" and r["source"] == "Yamazaki B9" and r["ratio_end"]):
        continue
    chi = int(r["chi"]); mu = float(r["mu"])
    if chi == 40 and mu == 1.25e-4:  # does not reach the gapped phase (q < 0.98; Paper 1b, App. A), excluded there too
        continue
    tag = "ab0" if mu == 0 else f"ab{mu:+.2e}"
    f = os.path.join(B, f"B9_K1.75_chi{chi}_{tag}", "levels.csv.gz")
    lv = {}
    for l in csv.DictReader(gzip.open(f, "rt")):
        lv.setdefault(int(l["step"]), []).append((l["sector"], float(l["x"])))
    ne = int(r["window_end"]); ns, Kn = [], []
    for n in range(12, ne + 1):
        L = lv[n]
        x01 = min(x for s, x in L if "(1, 2)" in s)
        xo = min((x for s, x in L if "(0, 1)" in s), key=lambda x: abs(x - K0))
        # solve x01 = 1/(4K)+aM g2, xo = K + aO g2 by Newton in K
        K = K0
        for _ in range(50):
            g2 = (xo - K) / aO
            F = 1/(4*K) + aM*g2 - x01; dF = -1/(4*K*K) - aM/aO
            K -= F/dF
        ns.append(n); Kn.append(K)
    ns, Kn = np.array(ns), np.array(Kn)
    sl, ic = np.polyfit(ns, Kn, 1); Ke = sl*ne + ic
    g2_end = float(r["g2_12"]) * (2**(2-K0)) ** (ne - 12)
    xm = float(r["xi_meas"]); xp0 = xi_pred(K0, ne, g2_end)
    g2e = g2_end * 4*K0*aM / (4*Ke*alpha(Ke, "m1"))
    xp1 = xi_pred(Ke, ne, g2e)
    rows.append(dict(chi=chi, mu=mu, n_end=ne, K_n12=Kn[0], K_end=Ke, dKdn=sl, ratio_input_K=xp0/xm, ratio_K_end=xp1/xm,
                     ratio_paper=float(r["ratio_end"])))
with open("kdrift.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for q in sorted(rows, key=lambda q: q["n_end"]):
    print(f"chi={q['chi']} mu={q['mu']:+.2e} n_end={q['n_end']} K(12)={q['K_n12']:.5f} K_end={q['K_end']:.5f} dK/dn={q['dKdn']:+.1e} "
          f"ratio: input K {q['ratio_input_K']:.3f} (paper {q['ratio_paper']:.3f}) -> K_end {q['ratio_K_end']:.3f}")
for lo, hi in ((56, 69), (70, 79), (80, 89)):
    s = [q for q in rows if lo <= q["n_end"] <= hi]
    if s:
        print(f"n_end {lo}-{hi}: {len(s)} runs, mean ratio input K {np.mean([q['ratio_input_K'] for q in s]):.3f}, with K_end {np.mean([q['ratio_K_end'] for q in s]):.3f}")
