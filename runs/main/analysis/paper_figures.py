"""Main-text figures and tables of main_en.tex from the single-environment runs (../data).

Usage:  python3 paper_figures.py      -> fig_*.pdf/png and paper_tables.tex in this folder
Fits follow docs/03_o2/3_1, 3_2, 3_4 and docs/04_weaker_symmetry/4_1, 4_2 (see analyze.py), with the
window variations used there: O(2) start 10/12/14 x cut |X| <= 0.05/0.10; U(1) last 8/10/12 points x
end shifted by 0/1/2 steps.  Ranges are min-max over the variations, not confidence intervals.
"""
import csv
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts (journals reject Type 3)
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from analyze import ALPHA, load, level, fit_growth

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
KS = (0.75, 1.0, 1.25, 1.5, 1.75, 2.5)
CHIS = (24, 32, 40, 48, 64)
COL = {24: "C0", 32: "C1", 40: "k", 48: "C2", 64: "C3"}
MK = {24: "s", 32: "^", 40: "o", 48: "v", 64: "D"}
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in"})
alpha = {round(float(r["K"]), 4): r for r in csv.DictReader(open(ALPHA))}


FOLLOW = os.path.join(HERE, "..", "..", "additional", "data")
KS_TERMS = (0.6,) + KS     # Fig. 2 also shows K = 0.6 (run B8), where umklapp and dimension-four terms separate


def run(sym, K, chi):
    if sym == "CU1" and abs(K - 0.6) < 1e-9:          # O(2), K = 0.6: follow-up run B8 (same environment)
        return load(os.path.join(FOLLOW, f"B8_K0.6_chi{chi}"))
    return load(os.path.join(DATA, f"{sym}_K{K:.4f}_chi{chi}"))


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + ".pdf"))
    fig.savefig(os.path.join(HERE, name + ".png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- O(2)
def o2_series(K, chi):
    lev, steps = run("CU1", K, chi)
    X, XJ = {}, {}
    for n, rows in lev.items():
        X[n] = 1 - 4 * K * level(rows, "(1, 2)", 1)
        v = sorted(float(r["x"]) for r in rows if "(0, 1)" in r["sector"] and int(r["rank"]) <= 3)
        if len(v) == 3:
            XJ[n] = 1 - min(((abs(v[i] - v[j]), (v[i] + v[j]) / 2) for i in range(3) for j in range(i + 1, 3)))[1]
    return X, XJ


def o2_fit(K, chi, ns=12, cut=0.10):
    X, XJ = o2_series(K, chi)
    out = [m for m in sorted(X) if m >= ns and abs(X[m]) > cut]
    if not out:
        return None
    ne = out[0] - 1
    n = np.arange(ns, ne + 1, dtype=float)
    _, lr, c, M = fit_growth(n, np.array([X[int(m)] for m in n]), ne, K < 0.75, K)
    cj, *_ = np.linalg.lstsq(M, np.array([XJ[int(m)] for m in n]), rcond=None)
    a = alpha[round(K, 4)]
    pred = float(a["J"]) / (4 * K * float(a["m1"]))
    return dict(ne=ne, r=math.exp(lr), q=(cj[2] / c[2]) / pred)


def o2_with_ranges(K, chi):
    main = o2_fit(K, chi)
    if main is None:
        return None
    var = [f for ns in (10, 12, 14) for cut in (0.05, 0.10) if (f := o2_fit(K, chi, ns, cut))]
    main["r_range"] = (min(f["r"] for f in var), max(f["r"] for f in var))
    main["q_range"] = (min(f["q"] for f in var), max(f["q"] for f in var))
    return main


def fig_terms():
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.9), sharey=True)
    for ax, K in zip(axes.flat, KS_TERMS):
        X, _ = o2_series(K, 40)
        big = [m for m in sorted(X) if m >= 6 and abs(X[m]) > 0.1]
        ne = big[0] - 1 if big else max(X)
        ns = 6
        n = np.arange(ns, ne + 1, dtype=float)
        x = np.array([X[int(m)] for m in n])
        t = (n - (ns + ne) / 2) / 10
        rates = {"C": -0.5, "D": 0.5}
        if K < 0.75:
            rates["B"] = 2.0 ** (2 - 4 * K)
        if K > 2:
            rates["A"] = 2.0 ** (2 - K)
            best = (None, None)
        def design(lr):
            cols, names = [np.ones_like(t), t], ["c0", "c1"]
            if K < 2:
                cols.append(np.exp(lr * (n - ne))); names.append("A")
            for k, lam in rates.items():
                cols.append(lam ** (n - ns)); names.append(k)
            return np.vstack(cols).T, names
        if K < 2:
            best = min(((np.linalg.lstsq(design(lr)[0], x, rcond=None)[1].sum() if len(x) > 6 else 0, lr)
                        for lr in np.linspace(math.log(1.02), math.log(4), 3000)))
            lr = best[1]
        else:
            lr = None
        M, names = design(lr)
        coef = dict(zip(names, np.linalg.lstsq(M, x, rcond=None)[0]))
        stop = next((m for m in sorted(X) if m >= 1 and abs(X[m]) > 0.3), max(X) + 1)
        nn = np.array([m for m in sorted(X) if 1 <= m < stop], float)
        bg = coef["c0"] + coef["c1"] * (nn - (ns + ne) / 2) / 10
        ax.semilogy(nn, np.abs(np.array([X[int(m)] for m in nn]) - bg), "k.", ms=3.5)
        terms = [("A", math.exp(lr) if lr else None, ne, "C3", r"$A\,r^{\,n-n_e}$ (dimerization)"),
                 ("B", rates.get("B"), ns, "C1", r"$A_{\rm u}(2^{2-4K})^{n-n_s}$ (umklapp$^2$)"),
                 ("C", -0.5, ns, "C0", r"$A_4(-1/2)^{n-n_s}$ (dim. 4, spin $\pm4$)"),
                 ("D", 0.5, ns, "C2", r"$A_0(1/2)^{n-n_s}$ (dim. 4, $T\bar T$)")]
        if K > 2:
            terms[0] = ("A", 2.0 ** (2 - K), ns, "C3", None)
        grid = np.arange(1, stop, 0.25)
        for key, lam, ref, col, lab in terms:
            if key not in coef or lam is None:
                continue
            y = abs(coef[key]) * abs(lam) ** (grid - ref)
            inside = (grid >= ns) & (grid <= ne)
            ax.semilogy(grid[inside], y[inside], "-", color=col, lw=1.1, label=lab)
            ax.semilogy(grid, y, ":", color=col, lw=0.8)
        ax.axhline(1e-4, color="0.4", ls="--", lw=0.7)
        ax.axvspan(ns, ne, color="0.92", zorder=0)
        ax.set_ylim(1e-6, 1)
        ax.set_xlim(0, stop)
        ax.set_title(rf"$K={K:g}$" + (rf",  $2^{{2-K}}={2 ** (2 - K):.2f}$" if K < 2 else ""), fontsize=8)
        if K == 0.6:
            handles, labels = ax.get_legend_handles_labels()
    leg = axes.flat[-1]                                  # the eighth panel holds the legend
    leg.axis("off")
    leg.legend(handles, labels, fontsize=6.5, loc="center", frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("RG step $n$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$|X-\beta_0-\beta_1 n|$")
    fig.tight_layout()
    save(fig, "fig_o2_terms_se")


def fig_rate_and_ratio(o2):
    for key, ylabel, name, ylim in (("r", r"$r/2^{2-K}$", "fig_o2_rate_se", (0.95, 1.05)),
                                    ("q", r"$(A_J/A)/[\alpha_J/(4K\alpha_{0,1})]$", "fig_o2_amp_se", (0.95, 1.08))):
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        ax.axhspan(0.97, 1.03, color="0.92", zorder=0)
        ax.axhline(1, color="k", lw=0.7)
        for j, chi in enumerate(CHIS):
            pts = [(K, o2[(K, chi)]) for K in KS if o2.get((K, chi))]
            dx = (j - 2) * 0.015
            for K, f in pts:
                norm = 2 ** (2 - K) if key == "r" else 1
                y = f[key] / norm
                if chi == 40:
                    lo, hi = (v / norm for v in f[key + "_range"])
                    ax.errorbar(K + dx, y, yerr=[[y - lo], [hi - y]], fmt=MK[chi], color=COL[chi], ms=4, capsize=2, lw=0.8)
                else:
                    ax.plot(K + dx, y, MK[chi], color=COL[chi], ms=3.5, mfc="none")
            ax.plot([], [], MK[chi], color=COL[chi], ms=3.5, mfc="white" if chi != 40 else COL[chi], label=rf"$\chi={chi}$")
        ax.set_xlabel("$K$"); ax.set_ylabel(ylabel); ax.set_ylim(*ylim); ax.set_xlim(0.65, 1.85)
        ax.legend(fontsize=6.5, ncol=3, frameon=False, loc="upper left")
        fig.tight_layout()
        save(fig, name)


# ---------------------------------------------------------------- U(1)
def u1_window(K, chi):
    lev, steps = run("U1", K, chi)
    good = []
    for m in sorted(lev):
        if m < 15:
            continue
        xp, xm = level(lev[m], "U₁](1)", 1), level(lev[m], "U₁](-1)", 1)
        c = float(steps[m]["central_charge"])
        if not (np.isfinite(xp) and np.isfinite(xm)) or max(abs(4 * K * xp - 1), abs(4 * K * xm - 1)) >= 0.1 \
                or abs(c - 1) >= 0.2:
            break
        good.append((m, (xp + xm) / 2, (xp - xm) / 2))
    return lev, good


def u1_fit(win):
    ne = win[-1][0]
    t = np.array([m - ne for m, *_ in win], float)
    o = np.array([w[2] for w in win])

    def resid(p):
        M = np.vstack([np.exp(p[0] * t) * np.cos(p[1] * t), np.exp(p[0] * t) * np.sin(p[1] * t)]).T
        uv, *_ = np.linalg.lstsq(M, o, rcond=None)
        return ((M @ uv - o) ** 2).sum(), uv
    starts = sorted((resid((a, b))[0], a, b) for a in np.linspace(-1.5, 1.5, 9) for b in np.linspace(0.01, math.pi - 0.01, 35))[:5]
    best = min((minimize(lambda p: resid(p)[0], (a, b), method="Nelder-Mead",
                         options=dict(xatol=1e-10, fatol=1e-22, maxiter=4000)) for _, a, b in starts), key=lambda r: r.fun)
    return dict(ne=ne, n_s=win[0][0], gamma=math.exp(best.x[0]), omega=math.degrees(best.x[1]), uv=resid(best.x)[1])


def u1_with_ranges(K, chi):
    lev, good = u1_window(K, chi)
    if len(good) < 10:
        return None
    main = u1_fit(good[-10:])
    var = [u1_fit(good[len(good) - L - s: len(good) - s]) for L in (8, 10, 12) for s in (0, 1, 2) if L + s <= len(good)]
    main["gamma_range"] = (min(v["gamma"] for v in var), max(v["gamma"] for v in var))
    main["windows"] = len(var)
    return main


def fig_u1_flows(u1):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    for ax, K, lab in zip(axes, (0.75, 1.5, 2.5), "abc"):
        lev, good = u1_window(K, 40)
        f = u1[(K, 40)]
        ns_plot = 15
        n = [m for m, *_ in good]
        ax.semilogy(n, [abs(e - 1 / (4 * K)) for _, e, _ in good], "s-", color="C0", ms=3, lw=0.7, label=r"$|E_1-1/(4K)|$")
        ax.semilogy(n, [abs(o) for *_, o in good], "o", color="C3", ms=3.2, label=r"$|\mathcal{O}_1|$")
        tt = np.arange(f["n_s"] - f["ne"], 1)
        u, v = f["uv"]; w = math.radians(f["omega"])
        ax.semilogy(f["ne"] + tt, np.abs(f["gamma"] ** tt * (u * np.cos(w * tt) + v * np.sin(w * tt))), "-", color="C3", lw=0.8, label=r"fit to $\mathcal{O}_1$")
        ax.axvspan(f["n_s"], f["ne"], color="#f6eeee", zorder=0)
        ax.set_title(f"({lab}) $K={K:g}$", fontsize=8.5, loc="left")
        ax.set_xlabel("RG step $n$"); ax.set_xlim(ns_plot - 0.5, f["ne"] + 0.5)
    axes[0].set_ylabel("level deviation"); axes[0].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_u1_flows_se")


def fig_u1_gamma(u1):
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.axhline(math.sqrt(2), color="k", ls="--", lw=0.7)
    for j, chi in enumerate(CHIS):
        dx = (j - 2) * 0.025
        for K in KS:
            f = u1.get((K, chi))
            if not f:
                continue
            if chi == 40:
                lo, hi = f["gamma_range"]
                ax.errorbar(K + dx, f["gamma"], yerr=[[f["gamma"] - lo], [hi - f["gamma"]]], fmt=MK[chi], color=COL[chi], ms=4, capsize=2, lw=0.8)
            else:
                ax.plot(K + dx, f["gamma"], MK[chi], color=COL[chi], ms=3.5, mfc="none")
        ax.plot([], [], MK[chi], color=COL[chi], ms=3.5, mfc="white" if chi != 40 else COL[chi], label=rf"$\chi={chi}$")
    ax.set_xlabel("$K$"); ax.set_ylabel(r"growth factor $\gamma$"); ax.set_ylim(1.400, 1.425)
    ax.set_yticks([1.40, 1.405, 1.41, math.sqrt(2), 1.42, 1.425]); ax.set_yticklabels(["1.400", "1.405", "1.410", r"$\sqrt{2}$", "1.420", "1.425"])
    ax.legend(fontsize=6.5, ncol=3, frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_u1_gamma_se")


# ---------------------------------------------------------------- K = 1 level repulsion
def k1_series(chi):
    lev, _ = run("CU1", 1.0, chi)
    S, x01 = {}, {}
    for n, rows in lev.items():
        v = sorted(float(r["x"]) for r in rows if "(1, 2)" in r["sector"])
        if len(v) >= 5:
            S[n] = (v[3] + v[4]) / 2 - (v[1] + v[2]) / 2
            x01[n] = v[0]
    return S, x01


def fig_k1():
    a01 = float(alpha[1.0]["m1"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    out = {}
    for chi in CHIS:
        S, x01 = k1_series(chi)
        fl = [n for n in S if 8 <= n <= 14]
        Sf, xf = np.median([S[n] for n in fl]), np.median([x01[n] for n in fl])
        ns = sorted(n for n in S if n >= 8)
        g2S = {n: (S[n] ** 2 - Sf ** 2) / (8 * math.pi ** 2) for n in ns}
        g2m = {n: (x01[n] - xf) / a01 for n in ns}
        win = [n for n in ns if 0.02 <= abs(1 - 4 * x01[n]) <= 0.10]
        ratio = [g2S[n] / g2m[n] for n in win]
        out[chi] = (min(ratio), max(ratio)) if ratio else None
        ax2.plot(win, ratio, MK[chi] + "-", color=COL[chi], ms=3, lw=0.7, label=rf"$\chi={chi}$")
        if chi == 40:
            stop = [n for n in ns if abs(1 - 4 * x01[n]) <= 0.3]
            ax1.semilogy(stop, [S[n] for n in stop], "o", color="C0", ms=3, label="splitting $S$")
            nn = np.array(stop, float)
            ax1.semilogy(nn, S[stop[-1]] * math.sqrt(2) ** (nn - stop[-1]), ":", color="C0", lw=0.8, label=r"$\propto(\sqrt{2})^n$")
            ax1.semilogy(stop, [abs(x01[n] - xf) for n in stop], "s", color="C1", ms=3, label=r"shift of $x_{0,1}$")
            ax1.semilogy(stop, [a01 * max(g2S[n], 1e-12) for n in stop], "-", color="C1", lw=0.9, label=r"$\alpha_{0,1}g_S^2$")
    ax1.set_title(r"(a) $K=1$, $\chi=40$", fontsize=8.5, loc="left"); ax1.set_xlabel("RG step $n$"); ax1.set_ylabel("level shift")
    ax1.set_ylim(1e-6, 1); ax1.legend(fontsize=6.5, frameon=False)
    ax2.axhline(1, color="0.4", ls="--", lw=0.7); ax2.set_ylim(0, 1.3)
    ax2.set_title(r"(b) $g_S^2/g_m^2$ where $x_{0,1}$ is shifted by 2–10%", fontsize=8.5, loc="left")
    ax2.set_xlabel("RG step $n$"); ax2.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "fig_k1_se")
    return out


# ---------------------------------------------------------------- drift of K_n
def fig_drift():
    rows = list(csv.DictReader(open(os.path.join(HERE, "drift_steps.csv"))))
    summ = list(csv.DictReader(open(os.path.join(HERE, "drift_summary.csv"))))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for chi in CHIS:
        r = [x for x in rows if float(x["K"]) == 2.5 and int(x["chi"]) == chi and int(x["n"]) >= 8]
        ax1.plot([int(x["n"]) for x in r], [float(x["K_n"]) - 2.5 for x in r], "-", color=COL[chi], lw=0.8, label=rf"$\chi={chi}$")
    ax1.set_xlabel("RG step $n$"); ax1.set_ylabel(r"$K_n-K$"); ax1.set_title(r"(a) $O(2)$, $K=2.5$", fontsize=8.5, loc="left")
    ax1.legend(fontsize=6.5, frameon=False, ncol=2)
    # K = 2.5 (no dimerization growth, whole run after n = 8) is the clean case; K = 1.25, 1.5 use the
    # shorter floor window before the dimerization term exceeds 1e-4 and are shown as open symbols.
    for K, mk, full in (("2.5", "D", True), ("1.25", "^", False), ("1.5", "s", False)):
        pts = [(int(s["chi"]), -float(s["dK_per_step"])) for s in summ if float(s["K"]) == float(K) and s.get("dK_per_step")
               and int(s["chi"]) >= 24 and -float(s["dK_per_step"]) > 0]
        ax2.loglog([p[0] for p in pts], [p[1] for p in pts], mk + ("-" if full else ":"), ms=3.5, lw=0.7,
                   mfc=None if full else "none", label=rf"$K={float(K):g}$")
        if full:
            sl, ic = np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)
            chi = np.array([24.0, 64.0])
            ax2.loglog(chi, np.exp(ic) * chi ** sl, "k--", lw=0.8, label=rf"fit $\propto\chi^{{{sl:.2f}}}$")
            print(f"K=2.5 direct drift exponent: {sl:.3f}")
    ax2.set_xticks(CHIS); ax2.set_xticklabels(CHIS); ax2.minorticks_off()
    ax2.set_xlabel(r"$\chi$"); ax2.set_ylabel(r"$-\,dK_n/dn$ (floor window)"); ax2.set_title("(b) direct drift of $K$", fontsize=8.5, loc="left")
    ax2.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    save(fig, "fig_drift_se")


def main():
    o2 = {(K, chi): o2_with_ranges(K, chi) for K in KS for chi in CHIS}
    u1 = {(K, chi): u1_with_ranges(K, chi) for K in KS for chi in CHIS}
    fig_terms(); fig_rate_and_ratio(o2); fig_u1_flows(u1); fig_u1_gamma(u1)
    k1 = fig_k1(); fig_drift()
    with open(os.path.join(HERE, "paper_tables.tex"), "w") as f:
        f.write("% O(2), chi = 40: K & window & r [range] & r/2^{2-K} & ratio [range]  (K = 0.6: run B8, same environment)\n")
        o2[(0.6, 40)] = o2_with_ranges(0.6, 40)
        for K in (0.6,) + KS:
            g = o2[(K, 40)]
            if g:
                f.write(f"{K:.2f} & 12--{g['ne']} & {g['r']:.3f} [{g['r_range'][0]:.3f}, {g['r_range'][1]:.3f}] & "
                        f"{g['r'] / 2 ** (2 - K):.3f} & {g['q']:.3f} [{g['q_range'][0]:.3f}, {g['q_range'][1]:.3f}] \\\\\n")
        f.write("% U(1), chi = 40: K & window & gamma [range] & omega\n")
        for K in KS:
            g = u1[(K, 40)]
            if g:
                f.write(f"{K:.2f} & {g['n_s']}--{g['ne']} & {g['gamma']:.5f} [{g['gamma_range'][0]:.5f}, {g['gamma_range'][1]:.5f}] & {g['omega']:.2f} \\\\\n")
        f.write("% K = 1 level repulsion: chi & min-max of g_S^2/g_m^2\n")
        for chi, rng in k1.items():
            f.write(f"% {chi}: {rng}\n")
    print(open(os.path.join(HERE, "paper_tables.tex")).read())


if __name__ == "__main__":
    main()
