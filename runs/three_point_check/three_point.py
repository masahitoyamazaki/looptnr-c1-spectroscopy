"""Three-point unsigned calibration test (paper 1b, question of 2026-11-25).

Could three full-length runs mu=0, +-mu0 read with the unsigned g_d^2(12) (protocol r_d=2^(2-K), |X|<=0.02)
have located mu* as well as the signed (pi-twisted) one-shot readout?  Existing data only.

Inputs
  Otsuki/sixvertex/data/sixvertex_sg_xi_predict.csv  (B9 rows, cut=0.02, rate=fixed): g2_12, window_end, n_e
  Yamazaki/runs/seed_refit_20261112/refit_b9_table.csv (full-scan global fit: rho, g_seed, mu*, rms, g2_closest)
  Yamazaki/runs/twist_campaign_summary_20261119/campaign_summary.csv (signed one-shot: n_cal, mu_pred, delay)

Three-point fit: ln g2 = 2 ln|rho (mu - m)| (rho>0) least squares over the three points, global in m
(profile cost over m on a dense sinh grid, all local minima refined).  The two lowest local minima are
reported: they correspond to the two sign branches (m inside or outside [-mu0, mu0]).  Also reported: vertex of
the exact interpolating quadratic in g2 (A + B mu + C mu^2), which has a unique solution but ignores the
constraint B^2 = 4AC.

Outputs: three_point_pairs.csv, three_point_mc.csv, run_length.csv, three_point_summary.txt
"""
import csv, math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "Otsuki/sixvertex/data/sixvertex_sg_xi_predict.csv"
FULL = ROOT / "Yamazaki/runs/seed_refit_20261112/refit_b9_table.csv"
CAMP = ROOT / "Yamazaki/runs/twist_campaign_summary_20261119/campaign_summary.csv"
# Table tab:signed (workstation of the main runs) where it differs from the campaign csv
SIGNED_TAB = {(1.0, 40): (30, 44), (1.25, 40): (38, 53), (1.5, 24): (39, 58), (1.5, 32): (45, 59), (1.5, 40): (48, 61)}
REFINED_NE = {(1.25, 40): 55, (1.5, 24): 58, (1.5, 32): 64, (1.5, 40): 71}  # tab:signed, refined counterterm


def ceiling_delay(key, R):
    """largest delay achieved by any run at this (K,chi): scan runs, signed one-shot, refined (empirical floor of the
    irregular component).  Parabola delays are capped at this value; the g2 floor is g2(mu=0)/2^((2-K) ceiling)."""
    ne0 = R[0.0]["ne"]
    best = max(r["ne"] for r in R.values()) - ne0
    c = camp.get(key)
    if c and c["mu_pred"] not in ("", "nan"):
        ne0s, nes = SIGNED_TAB.get(key, (int(c["ne_zero"]), int(c["ne_pred"])))
        best = max(best, nes - ne0s)
    if key in REFINED_NE:
        best = max(best, REFINED_NE[key] - SIGNED_TAB[key][0])
    return best

rows = [r for r in csv.DictReader(open(SRC)) if r["cut"] == "0.02" and r["rate"] == "fixed"
        and r["source"] == "Yamazaki B9" and r["g2_12"]]
runs = {}
for r in rows:
    runs.setdefault((float(r["K"]), int(r["chi"])), {})[float(r["mu"])] = dict(
        g2=float(r["g2_12"]), nend=int(r["window_end"]), ne=int(r["n_e"]) - 1)  # n_e: last step |X|<=0.1
full = {(float(t["K"]), int(t["chi"])): {k: float(v) for k, v in t.items()} for t in csv.DictReader(open(FULL))}
camp = {(float(c["K"]), int(c["chi"])): c for c in csv.DictReader(open(CAMP))}

T = np.sinh(np.linspace(-np.arcsinh(2000), np.arcsinh(2000), 40001))  # grid of m/mu0


def profile(lg, mu, m):
    """cost(m) and ln rho^2 for arrays: lg (...,3), mu (3,), m (..., G)."""
    d = lg[..., None, :] - 2 * np.log(np.abs(mu - m[..., None]) + 1e-300)
    c = d.mean(-1)
    return ((d - c[..., None]) ** 2).sum(-1), c


def fit3(mu0, g2m, g20, g2p):
    """global ln-LSQ through (-mu0,g2m),(0,g20),(+mu0,g2p).  Returns list of local minima (cost, m, rho), sorted."""
    mu = np.array([-mu0, 0.0, mu0]); lg = np.log([g2m, g20, g2p])
    m = T * mu0
    C, _ = profile(lg, mu, m)
    loc = np.flatnonzero((C[1:-1] <= C[:-2]) & (C[1:-1] <= C[2:])) + 1
    out = []
    for i in loc:
        r = minimize_scalar(lambda x: float(profile(lg, mu, np.array([x]))[0][0]), bounds=(m[i - 1], m[i + 1]),
                            method="bounded", options={"xatol": 1e-9 * mu0})
        c, lr = profile(lg, mu, np.array([r.x]))
        out.append((float(c[0]), float(r.x), math.exp(float(lr[0]) / 2)))
    out.sort()
    # keep distinct minima
    res = []
    for o in out:
        if all(abs(o[1] - p[1]) > 1e-6 * mu0 for p in res):
            res.append(o)
    return res


def vertex(mu0, g2m, g20, g2p):
    C = (g2p + g2m - 2 * g20) / (2 * mu0 ** 2); B = (g2p - g2m) / (2 * mu0)
    return -B / (2 * C) if C > 0 else float("nan")


def delay(F, K):
    return math.log(F) / math.log(2 ** (2 - K))


pair_rows, rl_rows, CEIL = [], [], {}
for key in sorted(runs):
    K, chi = key
    R = runs[key]; f = full[key]
    ms, rho, gs = f["mu_star"], f["rho"], f["g_seed"]
    g20 = R[0.0]["g2"]; ceil = ceiling_delay(key, R); floor = g20 / 2 ** ((2 - K) * ceil); CEIL[key] = (ceil, floor)
    pairs = sorted(m for m in R if m > 0 and -m in R)
    for mu0 in pairs:
        sols = fit3(mu0, R[-mu0]["g2"], g20, R[mu0]["g2"])
        c1, m1, r1 = sols[0]
        alt = [s for s in sols[1:] if np.sign(s[1]) != np.sign(m1) or (abs(s[1]) < mu0) != (abs(m1) < mu0)]
        c2, m2, r2 = alt[0] if alt else (float("nan"),) * 3
        g2res = (rho * (m1 - ms)) ** 2
        F_par = g20 / g2res
        # closest sampled run to mu*_3 (any mu in the scan)
        mc = min(R, key=lambda m: abs(m - m1))
        pair_rows.append(dict(
            K=K, chi=chi, mu_star=ms, mu0=mu0, mu0_over_mustar=mu0 / abs(ms),
            mu3=m1, rel_err=(m1 - ms) / abs(ms), rho3=r1, cost=c1,
            alt_mu3=m2, alt_cost=c2, cost_ratio=c2 / c1 if c1 > 0 else float("inf"),
            sign_ok=int(np.sign(m1) == np.sign(ms)), alt_signif=(c2 - c1) / f["rms_ln_g2"] ** 2,
            vertex=vertex(mu0, R[-mu0]["g2"], g20, R[mu0]["g2"]),
            vertex_rel_err=(vertex(mu0, R[-mu0]["g2"], g20, R[mu0]["g2"]) - ms) / abs(ms),
            g2_res_parab=g2res, F_parab=F_par, delay_parab=delay(F_par, K), delay_capped=min(delay(F_par, K), ceil), ceiling=ceil,
            mu_closest_sampled=mc, delay_closest_sampled=R[mc]["ne"] - R[0.0]["ne"],
            nend0=R[0.0]["nend"], nend_m=R[-mu0]["nend"], nend_p=R[mu0]["nend"],
            ne0=R[0.0]["ne"], ne_m=R[-mu0]["ne"], ne_p=R[mu0]["ne"],
            steps_unsigned=R[0.0]["nend"] + R[-mu0]["nend"] + R[mu0]["nend"]))

    c = camp.get(key)
    if c and c["mu_pred"] not in ("", "nan"):
        mp = float(c["mu_pred"]); ncal = int(c["n_cal"])
        g2s = (rho * (mp - ms)) ** 2
        ne0s, nes = SIGNED_TAB.get(key, (int(c["ne_zero"]), int(c["ne_pred"])))
        rl_rows.append(dict(K=K, chi=chi, n_cal=ncal, mu_pred=mp, signed_rel_err=(mp - ms) / abs(ms),
                            signed_delay_parab=delay(g20 / g2s, K), signed_delay_capped=min(delay(g20 / g2s, K), ceil),
                            signed_delay_meas=nes - ne0s, signed_ne=f"{ne0s}->{nes}",
                            nend0=R[0.0]["nend"], ne0=R[0.0]["ne"],
                            steps_signed_min=2 * ncal, steps_signed_7=7 * ncal))

with open(HERE / "three_point_pairs.csv", "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(pair_rows[0])); w.writeheader(); w.writerows(pair_rows)
with open(HERE / "signed_compare.csv", "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rl_rows[0])); w.writeheader(); w.writerows(rl_rows)

# ---------------- run length per (K,chi): unsigned three runs to n_end vs signed short runs to n_cal
ncal_of = {k: r["n_cal"] for k, r in ((( r["K"], r["chi"]), r) for r in rl_rows)}
rl = []
for p in pair_rows:
    k = (p["K"], p["chi"]); nc = ncal_of.get(k)
    rl.append(dict(K=p["K"], chi=p["chi"], mu0_over_mustar=p["mu0_over_mustar"], nend0=p["nend0"], nend_m=p["nend_m"],
                   nend_p=p["nend_p"], ne0=p["ne0"], steps_unsigned=p["steps_unsigned"], n_cal=nc,
                   ratio_vs_2ncal=p["steps_unsigned"] / (2 * nc), ratio_vs_7ncal=p["steps_unsigned"] / (7 * nc),
                   nend0_over_ncal=p["nend0"] / nc))
with open(HERE / "run_length.csv", "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rl[0])); w.writeheader(); w.writerows(rl)

# ---------------- Monte Carlo: noise sigma = rms of ln g2 about the full parabola
rng = np.random.default_rng(20261125)
U = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.2, 1.5, 2, 3, 5, 10, 20])
NS = 400
mc_rows = []


def fit3_fast(mu0, lg):
    """vectorized global fit over samples: lg (N,3) -> best m (N,), second-minimum flag."""
    mu = np.array([-mu0, 0.0, mu0])
    Tc = np.sinh(np.linspace(-np.arcsinh(500), np.arcsinh(500), 6001)) * mu0
    C, _ = profile(lg, mu, Tc[None, :])
    i = np.clip(C.argmin(1), 1, len(Tc) - 2)
    lo, hi = Tc[i - 1], Tc[i + 1]
    for _ in range(3):  # two successive refinements on a local grid
        g = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, 201)[None, :]
        Cg, _ = profile(lg, mu, g)
        j = np.clip(Cg.argmin(1), 1, 199)
        lo, hi = g[np.arange(len(g)), j - 1], g[np.arange(len(g)), j + 1]
    return 0.5 * (lo + hi)


for key in sorted(runs):
    K, chi = key; f = full[key]
    ms, rho, sig = f["mu_star"], f["rho"], f["rms_ln_g2"]; ceil, floor = CEIL[key]
    for u in U:
        mu0 = u * abs(ms)
        mu = np.array([-mu0, 0.0, mu0])
        lg0 = np.log((rho * (mu - ms)) ** 2 + floor)
        lg = lg0[None, :] + sig * rng.standard_normal((NS, 3))
        m3 = fit3_fast(mu0, lg)
        e = (m3 - ms) / abs(ms)
        ae = np.abs(e)
        g20 = (rho * ms) ** 2 + floor
        d = np.minimum(np.log(g20 / (rho * (m3 - ms)) ** 2) / math.log(2 ** (2 - K)), ceil)
        mc_rows.append(dict(K=K, chi=chi, sigma=sig, u=u, med_abs_err=np.median(ae), p84_abs_err=np.quantile(ae, 0.84),
                            frac_err_gt10=np.mean(ae > 0.1), frac_wrong_branch=np.mean(ae > 0.5),
                            frac_wrong_sign=np.mean(np.sign(m3) != np.sign(ms)),
                            med_delay=np.median(d), p16_delay=np.quantile(d, 0.16)))
with open(HERE / "three_point_mc.csv", "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(mc_rows[0])); w.writeheader(); w.writerows(mc_rows)

# ---------------- summary text
L = []
L.append("== data: symmetric pairs, mu*_3 (global ln-LSQ), alternative minimum, delays ==")
L.append(f"{'K':>4} {'chi':>3} {'mu0/|mu*|':>9} {'relerr':>8} {'vtx_err':>8} {'cost':>8} {'alt_mu3/mu*':>11} {'c2/c1':>8} {'dC/s2':>7} "
         f"{'d_par':>6} {'d_cap':>6} {'d_samp':>6} {'nend(-,0,+)':>12} {'ne(-,0,+)':>12}")
for p in pair_rows:
    L.append(f"{p['K']:4.2f} {p['chi']:3d} {p['mu0_over_mustar']:9.3f} {p['rel_err']:+8.3f} {p['vertex_rel_err']:+8.3f} "
             f"{p['cost']:8.1e} {p['alt_mu3'] / p['mu_star']:11.3f} {p['cost_ratio']:8.1e} {p['alt_signif']:7.1f} {p['delay_parab']:6.1f} "
             f"{p['delay_capped']:6.1f} {p['delay_closest_sampled']:6d} "
             f"{p['nend_m']:3d},{p['nend0']:3d},{p['nend_p']:3d}  {p['ne_m']:3d},{p['ne0']:3d},{p['ne_p']:3d}")
L.append("")
L.append("== signed one-shot (campaign / tab:signed) on the same parabola ==")
for s in rl_rows:
    L.append(f"{s['K']:4.2f} {s['chi']:3d} n_cal={s['n_cal']:2d} relerr={s['signed_rel_err']:+.3f} "
             f"delay parab={s['signed_delay_parab']:.1f} capped={s['signed_delay_capped']:.1f} measured={s['signed_delay_meas']} ({s['signed_ne']}) "
             f"| unsigned mu=0 run: n_end={s['nend0']} n_e={s['ne0']}; signed steps 2*n_cal={s['steps_signed_min']} 7*n_cal={s['steps_signed_7']}")
L.append("")
L.append("== Monte Carlo (noise sigma = rms ln g2 of full scan; g2 floor from the largest delay achieved at this (K,chi); delays capped there) ==")
for r in mc_rows:
    L.append(f"{r['K']:4.2f} {r['chi']:3d} sig={r['sigma']:.3f} u={r['u']:5.2f} med|err|={r['med_abs_err']:.3f} "
             f"p84={r['p84_abs_err']:.3f} P(>10%)={r['frac_err_gt10']:.2f} P(>50%)={r['frac_wrong_branch']:.2f} "
             f"P(sign)={r['frac_wrong_sign']:.2f} delay med={r['med_delay']:.1f} p16={r['p16_delay']:.1f}")
(HERE / "three_point_summary.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
