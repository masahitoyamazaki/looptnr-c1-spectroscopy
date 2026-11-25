"""Refit the compensation parabola of the alternating a/b deformation (B9 runs) with the seed read in the
protocol of the merged paper: rate fixed to 2^(2-K), window n>=12, |X|<=0.02 (columns g2_12 of
TNR/Otsuki/sixvertex/data/sixvertex_sg_xi_predict.csv, cut=0.02, rate=fixed).
Fit: ln g2(12) = 2 ln|g_seed + rho*mu| by least squares, global over mu* (fix of 2026-11-25, see global_fit); sign convention rho>0 (as in Table VI of the merged draft).
Output: refit_b9.csv
"""
import csv, math, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

SRC = Path(__file__).resolve().parents[3] / "Otsuki/sixvertex/data/sixvertex_sg_xi_predict.csv"
rows = [r for r in csv.DictReader(open(SRC)) if r["cut"] == "0.02" and r["rate"] == "fixed" and r["source"] == "Yamazaki B9" and r["g2_12"]]
groups = {}
for r in rows:
    groups.setdefault((float(r["K"]), int(r["chi"])), []).append((float(r["mu"]), float(r["g2_12"]), r))
def global_fit(mu, g2):
    """Global least squares of ln g2 = 2 ln|g_seed + rho*mu| (rho>0).
    2026-11-25 fix: the earlier version ran least_squares from 8 starting values only and landed in local
    minima for (K,chi)=(1.75,32), (1.25,40) and in several jackknife samples (see
    Yamazaki/runs/sec4_check_20261125/local_minima.txt).  Now: with g2 = rho^2 (mu-mu*)^2 the optimal ln rho at
    fixed mu* is the mean residual, so the cost is profiled over mu* on a dense grid (|mu*| <= 3 max|mu|),
    the 40 best grid minima are refined by bounded 1D minimization, and the best one is polished with
    least_squares in (g_seed, rho) (same residual and bounds as before; gives the Jacobian for the errors)."""
    lg = np.log(g2)

    def prof(ms):
        d = lg - 2 * np.log(np.abs(mu - ms) + 1e-300)
        c = d.mean()
        return float(np.sum((d - c) ** 2)), c

    span = 3 * np.max(np.abs(mu)) + 1e-6
    grid = np.linspace(-span, span, 600001)
    h = grid[1] - grid[0]
    Cc = []
    for chunk in np.array_split(grid, 60):
        D = lg[None, :] - 2 * np.log(np.abs(mu[None, :] - chunk[:, None]) + 1e-300)
        Cc.append(np.sum((D - D.mean(1, keepdims=True)) ** 2, 1))
    Cc = np.nan_to_num(np.concatenate(Cc), nan=np.inf)
    loc = np.flatnonzero((Cc[1:-1] <= Cc[:-2]) & (Cc[1:-1] <= Cc[2:])) + 1
    cand = sorted(zip(Cc[loc], grid[loc]))
    best = None
    for _, m0 in cand[:40]:
        r = minimize_scalar(lambda m: prof(m)[0], bounds=(m0 - h, m0 + h), method="bounded", options={"xatol": 1e-17})
        if best is None or r.fun < best.fun:
            best = r
    rho = math.exp(prof(best.x)[1] / 2)
    f = lambda p: 2 * np.log(np.abs(p[0] + p[1] * mu) + 1e-30) - np.log(g2)
    res = least_squares(f, [-rho * best.x, rho], bounds=([-1, 0], [1, 1e4]), x_scale=[abs(rho * best.x) + 1e-12, rho])
    assert 2 * res.cost <= best.fun * (1 + 1e-6) + 1e-12, (2 * res.cost, best.fun)
    return res


out = []
for (K, chi), pts in sorted(groups.items()):
    mu = np.array([p[0] for p in pts]); g2 = np.array([p[1] for p in pts])
    best = global_fit(mu, g2)  # 2026-11-25: global search (was: least_squares from 8 starts)
    gs, ka = best.x
    rms = math.sqrt(np.mean(best.fun ** 2))
    mu0 = [p for p in pts if p[0] == 0]
    out.append(dict(K=K, chi=chi, n=len(pts), rho=ka, g_seed=gs, mu_star=-gs / ka, rms_ln_g2=rms,
                    g2_mu0=mu0[0][1] if mu0 else float("nan")))
w = csv.DictWriter(open(Path(__file__).with_name("refit_b9.csv"), "w"), fieldnames=list(out[0]))
w.writeheader(); [w.writerow(o) for o in out]
for o in out:
    print(f"K={o['K']:.2f} chi={o['chi']:2d} n={o['n']:2d} rho={o['rho']:7.3f} g_seed={o['g_seed']:+.3e} mu*={o['mu_star']:+.3e} rms={o['rms_ln_g2']:.2f} g2(mu=0)={o['g2_mu0']:.2e}")

# ---- full table: run closest to mu*, reduction factor, measured and predicted delay
tab = []
for o in out:
    pts = groups[(o["K"], o["chi"])]
    p0 = [p for p in pts if p[0] == 0][0]
    pc = min((p for p in pts if p[0] != 0), key=lambda p: abs(p[0] - o["mu_star"]))
    F = p0[1] / pc[1]
    # the source tables give n_e as the first step with |X|>0.1; the paper (Table II) uses the last step with |X|<=0.1
    ne0, nec = int(p0[2]["n_e"]) - 1, int(pc[2]["n_e"]) - 1
    pred = math.log(F) / math.log(2 ** (2 - o["K"]))
    resid = {p[0]: 2 * math.log(abs(o["g_seed"] + o["rho"] * p[0])) - math.log(p[1]) for p in pts}
    tab.append(dict(o, mu_closest=pc[0], g2_closest=pc[1], F=F, ne0=ne0, ne_closest=nec, delay=nec - ne0, delay_pred=pred,
                    max_abs_resid=max(abs(v) for v in resid.values())))
w = csv.DictWriter(open(Path(__file__).with_name("refit_b9_table.csv"), "w"), fieldnames=list(tab[0]))
w.writeheader(); [w.writerow(t) for t in tab]
print()
for t in tab:
    print(f"{t['K']:.2f} & {t['chi']} & {t['rho']:.1f} & {t['g_seed']:+.1e} & {t['mu_star']:.2e} & {t['g2_mu0']:.1e} & {t['g2_closest']:.1e} (mu={t['mu_closest']:.1e}) & {t['ne0']}->{t['ne_closest']} & {t['delay']} ({t['delay_pred']:.1f})  F={t['F']:.0f} maxres={t['max_abs_resid']:.2f}")

# ---- figure in the style of additional/analysis/b9.py, with the unified readout
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "mathtext.fontset": "cm"})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
Ks = sorted({o["K"] for o in out}); cmap = plt.get_cmap("viridis")
for o in out:
    K, chi = o["K"], o["chi"]
    if chi == 24:
        continue
    pts = sorted(groups[(K, chi)], key=lambda p: p[0])
    p0 = [p for p in pts if p[0] == 0][0]
    col = cmap(Ks.index(K) / max(len(Ks) - 1, 1)); mk = "o" if chi == 32 else "s"
    u = np.array([p[0] / o["mu_star"] for p in pts])
    ax[0].plot(u, [p[1] / p0[1] for p in pts], mk, ms=3, color=col, mfc="none" if chi == 40 else col,
               label=rf"$K={K:g}$" if chi == 32 else None)
    ax[1].plot(u, [int(p[2]["n_e"]) - int(p0[2]["n_e"]) for p in pts], mk + "-", ms=3, lw=0.6, color=col,
               mfc="none" if chi == 40 else col)
uu = np.linspace(-1.2, 2.2, 400)
ax[0].plot(uu, (1 - uu) ** 2, "k-", lw=0.8, label=r"$(1-\mu/\mu^*)^2$")
ax[0].set_yscale("log"); ax[0].set_ylim(1e-4, 30); ax[0].set_xlim(-1.2, 2.2)
ax[0].set_xlabel(r"$\mu/\mu^*$"); ax[0].set_ylabel(r"$g_{\rm d}^2(12)\,/\,g_{\rm d}^2(12)|_{\mu=0}$")
ax[0].legend(fontsize=6, frameon=False, ncol=2); ax[0].set_title(r"(a) seed (filled: $\chi=32$, open: $\chi=40$)", fontsize=8.5)
ax[1].axvline(1, color="k", ls=":", lw=0.8); ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_xlim(-1.2, 2.2); ax[1].set_xlabel(r"$\mu/\mu^*$"); ax[1].set_ylabel(r"$n_e-n_e|_{\mu=0}$")
ax[1].set_title(r"(b) delay of the breakdown", fontsize=8.5)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(Path(__file__).with_name(f"fig_b9_unified.{ext}"), dpi=200)
