"""Referee round 5, majors 1(d) and 3.  For the deformation runs (followup_B B9 scans, chi = 32, 40) at all K:
(i)  K_n per step from the lowest charged level and one level of the e=+-1 pair with their second-order shifts
     (C-odd level; C-even at K=1, where the C-odd one is degenerate with the current), linear fit over the window;
     the prediction of Eq. (12) re-evaluated with K_eff = K_n at the window centre, at n_end, and extrapolated to the
     last step used for xi (n_last).
(ii) At K = 1.75: the window fit of Eq. (6) refitted with an additional cross term B r_d^{(n-n_end)/2} (a source in
     Eq. (5) adds g_d^2 ~ a^2 r^(2n) + 2 a c r^n), (a) with B free and (b) with B fixed by the measured source:
     g_d(n) = A12(mu) r^(n-12) + c, c = -f12 A12(0) from the campaign fit at the same chi (campaign_summary.csv),
     A12(mu)/A12(0) = (g_seed + rho mu)/g_seed (Table II), so B/A = 2 c /(A12(mu) r^(n_end-12)).
Usage: python kdrift_all.py <root of the collaboration repository>.  Output: kdrift_all.csv, crossterm_K175.csv"""
import csv, gzip, math, os, sys, numpy as np
T = sys.argv[1]
D = os.path.join(T, "Otsuki/sixvertex/data"); B = os.path.join(T, "Yamazaki/runs/additional/data")
al = {round(float(r["K"]), 3): r for r in csv.DictReader(open(os.path.join(D, "sixvertex_g2_alpha.csv")))}
Ks = sorted(al)
alpha = lambda K, key: float(np.interp(K, Ks, [float(al[k][key]) for k in Ks]))
def kappa_sg(K):
    p = K / (2 - K)
    return (math.gamma(p/(p+1)) / math.gamma(1/(p+1)) / math.pi
            * (math.sqrt(math.pi)*math.gamma((p+1)/2) / (2*math.gamma(p/2))) ** (2/(p+1)))
def xi_pred(K, n, g2):
    return 2*math.sqrt(2)*math.sqrt(2)**n/(2*math.pi) * (math.sqrt(2)*kappa_sg(K)/math.sqrt(g2)) ** (1/(2-K))
def levels(K, chi, mu):
    tag = "ab0" if mu == 0 else f"ab{mu:+.2e}"
    f = os.path.join(B, f"B9_K{K}_chi{chi}_{tag}", "levels.csv.gz")
    if not os.path.exists(f):
        tag = "ab0" if mu == 0 else f"ab{mu:+g}"
        f = os.path.join(B, f"B9_K{K}_chi{chi}_{tag}", "levels.csv.gz")
    if not os.path.exists(f):
        return None
    lv = {}
    for l in csv.DictReader(gzip.open(f, "rt")):
        lv.setdefault(int(l["step"]), []).append((l["sector"], float(l["x"])))
    return lv
CS = os.path.join(T, "Yamazaki/runs/twist_campaign_summary_20261119/campaign_summary.csv")
f12 = {int(q["chi"]): float(q["f12"]) for q in csv.DictReader(open(CS)) if q["K"] == "1.75"}
B9 = {int(q["chi"]): (float(q["g_seed"]), float(q["rho"])) for q in csv.DictReader(open(os.path.join(T, "Yamazaki/runs/seed_refit_20261112/refit_b9_table.csv"))) if q["K"] == "1.75"}
out, cross = [], []
for r in csv.DictReader(open(os.path.join(D, "sixvertex_sg_xi_predict.csv"))):
    if not (r["cut"] == "0.02" and r["rate"] == "fixed" and r["source"] == "Yamazaki B9" and r["ratio_end"]):
        continue
    K0 = float(r["K"]); chi = int(r["chi"]); mu = float(r["mu"])
    if K0 == 1.75 and chi == 40 and mu == 1.25e-4:   # does not reach the gapped phase (Paper 1b, App. A)
        continue
    lv = levels(r["K"], chi, mu)
    if lv is None:
        print("missing", r["run"]); continue
    sec = "(0, 0)" if K0 == 1.0 else "(0, 1)"; key = "e1E" if K0 == 1.0 else "e1O"
    aM, aE = alpha(K0, "m1"), alpha(K0, key)
    ne = int(r["window_end"]); nl = int(r["n_last"]); ns, Kn, X = [], [], []
    for n in range(12, ne + 1):
        L = lv[n]; x01 = min(x for s, x in L if "(1, 2)" in s)
        xe = min((x for s, x in L if sec in s and x > 0.3), key=lambda x: abs(x - K0))
        K = K0
        for _ in range(60):
            g2 = (xe - K) / aE; F = 1/(4*K) + aM*g2 - x01; dF = -1/(4*K*K) - aM/aE; K -= F/dF
        ns.append(n); Kn.append(K); X.append(1 - 4*K0*x01)
    ns, Kn, X = np.array(ns), np.array(Kn), np.array(X)
    sl, ic = np.polyfit(ns, Kn, 1)
    g2_end = float(r["g2_12"]) * (2**(2-K0)) ** (ne - 12); xm = float(r["xi_meas"])
    def ratio(Ke):
        return xi_pred(Ke, ne, g2_end * 4*K0*aM / (4*Ke*alpha(Ke, "m1"))) / xm
    row = dict(K=K0, chi=chi, mu=mu, n_end=ne, n_last=nl, dKdn=sl, K_mid=sl*(12+ne)/2+ic, K_end=sl*ne+ic, K_last=sl*nl+ic,
               ratio_input=ratio(K0), ratio_mid=ratio(sl*(12+ne)/2+ic), ratio_end=ratio(sl*ne+ic), ratio_last=ratio(sl*nl+ic))
    out.append(row)
    if K0 == 1.75:   # (ii) window refit with a cross term, same window as the deposited fit
        rd = 2 ** (2 - K0); w = np.abs(X) <= 0.02; nn, xx = ns[w], X[w]
        A0 = np.vstack([np.ones_like(nn), nn, rd ** (nn - ne)]).T
        A1 = np.column_stack([A0, rd ** ((nn - ne) / 2)])
        c0 = np.linalg.lstsq(A0, xx, rcond=None)[0]; c1 = np.linalg.lstsq(A1, xx, rcond=None)[0]
        g0 = -c0[2] / (4*K0*aM); g1 = -c1[2] / (4*K0*aM)
        gs, rho = B9[chi]; r = math.sqrt(2) ** (2 - K0)
        BA = 2 * (-f12[chi]) / (((gs + rho * mu) / gs) * r ** (ne - 12))
        A2 = np.vstack([np.ones_like(nn), nn, rd ** (nn - ne) + BA * rd ** ((nn - ne) / 2)]).T
        c2 = np.linalg.lstsq(A2, xx, rcond=None)[0]; g2f = -c2[2] / (4*K0*aM)
        cross.append(dict(chi=chi, mu=mu, n_end=ne, ratio_no_cross=xi_pred(K0, ne, g0)/xm if g0 > 0 else np.nan,
                          ratio_cross=xi_pred(K0, ne, g1)/xm if g1 > 0 else np.nan, cross_over_growth_at_end=c1[3]/c1[2],
                          BA_fixed=BA, ratio_cross_fixed=xi_pred(K0, ne, g2f)/xm if g2f > 0 else np.nan))
for name, rows in (("kdrift_all.csv", out), ("crossterm_K175.csv", cross)):
    with open(name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for K in sorted({q["K"] for q in out}):
    s = [q for q in out if q["K"] == K]
    med = lambda k: np.median([q[k] for q in s])
    print(f"K={K}: {len(s)} runs, dK/dn {min(q['dKdn'] for q in s):+.1e}..{max(q['dKdn'] for q in s):+.1e}; median ratio input {med('ratio_input'):.3f}, "
          f"K_mid {med('ratio_mid'):.3f}, K_end {med('ratio_end'):.3f}, K_last {med('ratio_last'):.3f}")
    if K == 1.75:
        for lo, hi in ((56, 69), (70, 79), (80, 89)):
            t = [q for q in s if lo <= q["n_end"] <= hi]
            print(f"   n_end {lo}-{hi} ({len(t)}): " + ", ".join(f"{k} {np.mean([q[k] for q in t]):.3f}" for k in ("ratio_input", "ratio_mid", "ratio_end", "ratio_last")))
for lo, hi in ((56, 69), (70, 79), (80, 89)):
    t = [q for q in cross if lo <= q["n_end"] <= hi]
    print(f"cross term K=1.75 n_end {lo}-{hi} ({len(t)}): no cross {np.nanmean([q['ratio_no_cross'] for q in t]):.3f}, with cross {np.nanmean([q['ratio_cross'] for q in t]):.3f}, "
          f"B/A at end {np.median([q['cross_over_growth_at_end'] for q in t]):+.3f}; fixed B/A {min(q['BA_fixed'] for q in t):+.3f}..{max(q['BA_fixed'] for q in t):+.3f} -> {np.nanmean([q['ratio_cross_fixed'] for q in t]):.3f}")
print("fixed cross term: median ratio %.3f (input-K fit %.3f)" % (np.nanmedian([q['ratio_cross_fixed'] for q in cross]), np.nanmedian([q['ratio_no_cross'] for q in cross])))
