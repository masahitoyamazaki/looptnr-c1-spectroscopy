"""All figures of paper 2 in the style of paper 1 (serif, Computer Modern math, 8.5 pt, Type-42 fonts, PDF).
No tensor computation; inputs listed in common.py. Adapted from Otsuki's sixvertex_sg_xi_figs.py and
sixvertex_phase_figs.py and from Yamamoto's report10_correlation_length.py (same data, same definitions).

Usage (from paper2_draft/): ~/miniforge3/bin/python scripts/make_figures.py
"""
import csv
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from common import (CUT, FIG, KCOL, KMARK, KS, OTS, YAM, alpha01, ell, kappa_K, predict_rows, read, runs,
                    style, xi_from_g)

style()
ROWS = predict_rows()
W1, W2 = 3.4, 7.0


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------------------------------------
def fig_xi_vs_g():
    fig, ax = plt.subplots(figsize=(W1, 2.7))
    for K in KS:
        sel = runs(ROWS, K)
        g = np.array([math.sqrt(r["g2_12"]) for r in sel])
        gs = np.logspace(math.log10(g.min()) - 0.3, math.log10(g.max()) + 0.3, 50)
        ax.plot(gs, [xi_from_g(K, 12, v) for v in gs], "--", lw=0.7, color=KCOL[K])
        ax.plot(g, [r["xi_meas"] for r in sel], KMARK[K], color=KCOL[K], mfc="none", mew=0.7, label=f"$K={K:g}$")
    ax.set(xscale="log", yscale="log", xlabel=r"$|g_{\rm d}(12)|$", ylabel=r"measured $\xi/a_0$")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_xi_vs_g")


def fig_ratio():
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5), sharey=True)
    for ax, cut, title in zip(axes, (0.10, 0.02), (r"(a) $|X|\leq0.1$", r"(b) $|X|\leq0.02$")):
        for K in KS:
            sel = runs(ROWS, K, cut)
            main = [r for r in sel if r["chi"] != 16]
            s16 = [r for r in sel if r["chi"] == 16]
            ax.plot([r["xi_meas"] for r in main], [r["ratio_end"] for r in main], KMARK[K], color=KCOL[K],
                    label=f"$K={K:g}$ ({len(sel)})")
            ax.plot([r["xi_meas"] for r in s16], [r["ratio_end"] for r in s16], KMARK[K], color=KCOL[K], mfc="none",
                    mew=0.7)
        ax.axhline(1, color="k", lw=0.6)
        ax.set(xscale="log", xlabel=r"measured $\xi/a_0$", title=title)
    axes[0].set_ylabel(r"$\xi_{\rm pred}/\xi$")
    axes[0].set_ylim(0.5, 1.15)
    axes[1].legend(fontsize=6.5, loc="lower left", ncol=2)
    fig.tight_layout()
    save(fig, "fig_ratio")


def fig_collapse():
    levels = read(OTS / "sixvertex_sg_xi_levels.csv")
    steps = read(OTS / "sixvertex_sg_xi_predict_steps.csv")
    Ks = (1.0, 1.25, 1.5, 1.75)
    fig, axes = plt.subplots(2, 4, figsize=(W2, 3.9), sharex="col")
    cmap = plt.get_cmap("Blues")
    for j, K in enumerate(Ks):
        axX, axM = axes[0, j], axes[1, j]
        sel = sorted([r for r in runs(ROWS, K) if r["source"] == "Yamazaki B9" and r["chi"] == 40],
                     key=lambda r: r["xi_meas"])
        x0 = 10 ** math.floor(math.log10(min(ell(12) / r["xi_pred_end"] for r in sel)) + 0.3)
        t = np.logspace(math.log10(x0) - 0.5, 1.6, 300)
        for i, r in enumerate(sel):
            c = cmap(0.35 + 0.6 * i / max(len(sel) - 1, 1))
            used = {int(q["n"]) for q in steps if q["run"] == r["run"] and q["cut"] == CUT and q["rate"] == "fixed"}
            pts = [v for v in levels if v["run"] == r["run"] and v["n"] >= 12]
            Lx = np.array([v["ell"] / r["xi_pred_end"] for v in pts])
            inside = np.array([int(v["n"]) in used for v in pts])
            for ax, ys, ls in ((axX, [abs(v["X"]) for v in pts], "-"), (axM, [v["ratio_C_odd"] for v in pts], "-"),
                               (axM, [v["ratio_C_even"] for v in pts], "--")):
                ys = np.array(ys)
                ax.plot(Lx, ys, ls, color=c, lw=0.5)
                ax.plot(Lx[~inside], ys[~inside], ".", ms=2.2, color=c)
                ax.plot(Lx[inside], ys[inside], "o", ms=2.2, mfc="white", mec=c, mew=0.5)
        weak = 4 * K * alpha01(K) * 2 * kappa_K(K) ** 2 * (t / (2 * math.pi)) ** (2 * (2 - K))
        axX.plot(t, weak, "k--", lw=0.7)
        axX.plot(t, 4 * K * t / (2 * math.pi), "k:", lw=0.7)
        axX.axhline(CUT, color="0.5", lw=0.6, ls="-.")
        axX.set(yscale="log", ylim=(1e-6, 30), title=f"$K={K:g}$ ({len(sel)} runs)")
        axM.axhline(4 * K * K, color="0.5", lw=0.6, ls=":")
        axM.axhline(2, color="C3", lw=0.6, ls=":")
        axM.set(yscale="log", ylim=(1.5, 16))
        for ax in (axX, axM):
            ax.set_xscale("log")
            ax.set_xlim(x0, 40)
        axM.set_xlabel(r"$L/\xi_{\rm pred}$")
    axes[0, 0].set_ylabel(r"$|X|$")
    axes[1, 0].set_ylabel("mass ratio")
    fig.tight_layout(h_pad=0.3, w_pad=0.4)
    save(fig, "fig_collapse")


def phase_levels(tag):
    out = {}
    try:
        with open(OTS / f"sixvertex_phase_{tag}_levels.csv", encoding="utf8") as f:
            for r in csv.DictReader(f):
                out.setdefault(int(r["step"]), {}).setdefault(r["sector"], []).append(float(r["ratio"]))
    except FileNotFoundError:
        pass
    return out


MASS_RUNS = [(0.9, {"std": ["sg_K0.9"], "wide": ["sg10_K0.9"]}),
             (0.75, {"std": ["sg_K0.75_early", "sg_K0.75"], "wide": ["sg10_K0.75"]}),
             (0.6, {"std": ["sg_K0.6_early", "sg_K0.6"], "wide": ["sg10_K0.6"]})]
TUBES = {"std": (lambda n: ell(n), lambda n: ell(n) / 2, 1e-7),
         "wide": (lambda n: 2 * math.sqrt(10) * math.sqrt(2) ** n, lambda n: 4 / math.sqrt(10) * math.sqrt(2) ** n, 1e-4)}


def mass_series(tags, floor):
    data = {}
    for tag in tags:
        data.update(phase_levels(tag))
    top = lambda n, sec, i: sorted(data[n][sec], reverse=True)[i]
    used = (("(0;1)", 0), ("(0;0)", 1), ("(1;2)", 0))
    ns = []
    for n in sorted(data):
        if not all(len(data[n].get(s, [])) > i and top(n, s, i) > floor for s, i in used):
            break
        ns.append(n)
    return ns, (lambda n, sec, i: -math.log(top(n, sec, i)))


def fig_mass():
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.3), sharey=True)
    for ax, (K, tubes), lab in zip(axes, MASS_RUNS, "abc"):
        p = K / (2 - K)
        xi = None
        for tube, tags in tubes.items():
            Lf, hf, floor = TUBES[tube]
            ns, x = mass_series(tags, floor)
            if tube == "std":
                xi = sum(hf(n) / x(n, "(1;2)", 0) for n in ns[-2:]) / 2
            filled = tube == "std"
            for sec, i, col, mk in (("(0;1)", 0, "C0", "o"), ("(0;0)", 1, "C1", "s")):
                ax.plot([Lf(n) for n in ns], [x(n, sec, i) / x(n, "(1;2)", 0) for n in ns], mk + ("-" if filled else "--"),
                        color=col, lw=0.7, ms=2.5, mfc=col if filled else "white", mew=0.6)
        ax.axhline(2 * math.sin(math.pi * p / 2), color="C0", ls=":", lw=0.8)
        ax.axhline(2 * math.sin(math.pi * p) if 2 * p < 1 else 2, color="C1", ls=":", lw=0.8)
        ax.axhline(4 * K * K, color="0.5", ls=":", lw=0.8)
        ax.axvline(xi, color="0.5", ls="--", lw=0.7)
        ax.set(xscale="log", xlabel=r"$L/a_0$", title=f"({lab}) $K={K:g}$")
        print(f"mass K={K}: xi={xi:.4g}")
    axes[0].set_ylabel(r"mass ratio to the $|m|=1$ soliton")
    fig.tight_layout()
    save(fig, "fig_mass")


def fig_mu():
    K = 0.75
    sel = runs(ROWS, K)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.6))
    gs = np.logspace(-6.5, -3, 100)
    a1.plot(gs, [xi_from_g(K, 12, g) for g in gs], "k--", lw=0.7)
    groups = (("this plan", 40, "C0", "o", r"$a/b$, $\chi=40$ (separate)"),
              ("Yamazaki B9", 40, "C1", "^", r"$a/b$, $\chi=40$"),
              ("Yamazaki B9", 32, "C2", "s", r"$a/b$, $\chi=32$"),
              ("Yamazaki chi scan", None, "C3", "D", r"$\mu=0$, $\chi=16$--$64$"))
    for src, chi, c, m, lab in groups:
        q = [r for r in sel if r["source"] == src and (chi is None or r["chi"] == chi)]
        a1.plot([math.sqrt(r["g2_12"]) for r in q], [r["xi_meas"] for r in q], m, color=c, mfc="none", mew=0.7, label=lab)
    a1.set(xscale="log", yscale="log", xlabel=r"$|g_{\rm d}(12)|$", ylabel=r"measured $\xi/a_0$", title=r"(a) $K=0.75$")
    a1.legend(fontsize=6.5, loc="upper right")
    pts = [r for r in sel if r["chi"] == 40 and r["source"] in ("this plan", "Yamazaki B9")]
    mus = np.array([r["mu"] for r in pts])
    lg2 = np.array([math.log(r["g2_12"]) for r in pts])
    cost = lambda v: float(np.sum((lg2 - np.log((v[1] * 1e-5 + v[0] * mus) ** 2)) ** 2))
    res = minimize(cost, [29.4, -4.5], method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-14, maxiter=20000))
    kf, gf = res.x[0], res.x[1] * 1e-5
    print(f"mu fit K=0.75 chi=40: kappa={kf:.3g}, g_seed={gf:.3g}, mu*={-gf / kf:.3g}, {len(pts)} runs")
    ml = np.linspace(-6e-5, 6e-5, 2001)
    a2.plot(ml * 1e6, [xi_from_g(K, 12, gf + kf * m) for m in ml], "k--", lw=0.7, label="fit, present protocol")
    a2.plot(ml * 1e6, [xi_from_g(K, 12, -4.5e-5 + 29.4 * m) for m in ml], ":", color="0.5", lw=0.8, label="Ref. [Paper 1], Table VI")
    for src, chi, c, m, lab in groups[:2]:
        q = [r for r in pts if r["source"] == src]
        a2.plot([r["mu"] * 1e6 for r in q], [r["xi_meas"] for r in q], m, color=c, mfc="none", mew=0.7, label=lab)
    a2.set(yscale="log", ylim=(1e3, 2e6), xlabel=r"$\mu\ [10^{-6}]$", ylabel=r"measured $\xi/a_0$", title=r"(b) $K=0.75$, $\chi=40$")
    a2.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_mu_K075")


def fig_bkt():
    comp = {(r["K"], int(r["chi"])): r for r in read(YAM / "comparison.csv")}
    prof = [r for r in read(YAM / "profiles.csv") if r["K"] == 0.5]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.5))
    chis = (32, 40, 48, 64)
    for j, chi in enumerate(chis):
        pp = sorted([r for r in prof if int(r["chi"]) == chi], key=lambda r: r["step"])
        c = f"C{j}"
        a1.plot([r["step"] for r in pp], [r["xi_m"] for r in pp], "-", color=c, lw=0.8, label=rf"$\chi={chi}$")
        a1.axhline(comp[(0.5, chi)]["xi_pred"], color=c, ls=":", lw=0.7)
    a1.set(yscale="log", xlabel="RG step $n$", ylabel=r"$\xi_m/a_0$", title=r"(a) $K=1/2$")
    a1.legend(fontsize=6.5, loc="lower right")
    rr = [comp[(0.5, c)] for c in chis]
    a2.plot(chis, [r["xi_measured"] / r["xi_no_log"] for r in rr], "s-", color="0.5", lw=0.7, ms=3, label="without logarithmic running")
    a2.plot(chis, [r["xi_measured"] / r["xi_pred"] for r in rr], "o-", color="C0", lw=0.7, ms=3, label="with two-loop running")
    a2.axhline(1, color="k", lw=0.6)
    a2.set(xlabel=r"bond dimension $\chi$", ylabel=r"$\xi/\xi_{\rm pred}$", xticks=chis, title="(b) effect of the logarithm")
    a2.legend(fontsize=6.5, loc="center right")
    fig.tight_layout()
    save(fig, "fig_bkt")


def fig_xi_vs_chi():
    rows = [r for r in ROWS if r["cut"] == CUT and r["rate"] == "fixed" and r["mu"] == 0 and r["settled"] == 1
            and r["source"] in ("Yamazaki chi scan", "Yamazaki B2", "Yamazaki B5")]
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    for K in (0.75, 1.0, 1.25, 1.5):
        rs = sorted([r for r in rows if r["K"] == K], key=lambda r: r["chi"])
        chi = [r["chi"] for r in rs]
        ax.plot(chi, [r["xi_meas"] for r in rs], KMARK[K], color=KCOL[K], mfc="none", mew=0.7, label=f"$K={K:g}$")
        ax.plot(chi, [r["xi_pred_end"] for r in rs], "-", lw=0.6, color=KCOL[K])
    ax.set(yscale="log", xticks=[16, 24, 32, 40, 48, 64, 96], xlabel=r"bond dimension $\chi$", ylabel=r"$\xi/a_0$")
    ax.legend(fontsize=6.5, loc="lower right", ncol=2)
    fig.tight_layout()
    save(fig, "fig_xi_vs_chi")


if __name__ == "__main__":
    fig_xi_vs_g()
    fig_ratio()
    fig_collapse()
    fig_mass()
    fig_mu()
    fig_bkt()
    fig_xi_vs_chi()
