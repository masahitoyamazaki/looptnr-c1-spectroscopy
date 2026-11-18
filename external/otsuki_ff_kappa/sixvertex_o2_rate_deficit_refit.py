# Is the deficit of the O(2) growth rate for K <= 1 (paper 1a, L457-462) the fourth-order (g_d^4) term of the level?
#
# The charged level X(n) = 1 - 4K x_01(n) receives -4K alpha g_d^2 + O(g_d^4). Writing the growth part as
# X_g = A r^{n - n_e} with r = 2^{2-K}, the g_d^4 term is kappa X_g^2 = kappa A^2 r^{2(n - n_e)}: its rate r^2 is fixed
# by perturbation theory, only kappa is unknown. A saturation (kappa X_g < 0 relative to X_g, i.e. kappa > 0 for A < 0)
# lowers the apparent rate of a single-exponential fit and the more so the wider the window, as the paper observes.
#
# A (saved data, no new RG): growth window of Table IV (n >= 12, |X| <= cut), three models with the same background
#   G0 (paper): c + d t + A r^{n-n_e}, r free                                   [+ umklapp (2^{2-4K})^{n-n_s} for K < 3/4]
#   G1:         c + d t + A r^{n-n_e} + B r^{2(n-n_e)}, r = 2^{2-K} fixed        (same number of parameters as G0)
#   G2:         as G1 but r free
#   kappa = B / A^2 from G1 should be independent of chi and of the window if the deficit is this term.
# B (prediction at K = 1): kappa from the exact free-fermion scaling function (sixvertex_ff_dimerization_scaling.py),
#   and a synthetic X(n) = F(s_n), s_n proportional to 2^{n/2}, fitted with the protocol of G0 (as the synthetic flow
#   of paper App. k175), giving the predicted r/2^{2-K} at K = 1 for each window.
#
# Data (read only): as sixvertex_o2_j3jbar_refit.py (Yamazaki/runs/cpu_baseline_20261106, followup_B_20261106 B8).
# Usage (from the repository root):
#   uv run --no-project --with numpy --with scipy --with matplotlib python Otsuki/sixvertex/scripts/sixvertex_o2_rate_deficit_refit.py
# Outputs: Otsuki/sixvertex/data/sixvertex_o2_rate_deficit_{fits,synthetic}.csv,
#          Otsuki/sixvertex/figure/sixvertex_o2_rate_deficit_{kappa,k1}.png,
#          Otsuki/sixvertex/logs/sixvertex_o2_rate_deficit_report.log

import csv
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sixvertex_o2_j3jbar_refit as rf                      # noqa: E402  (data loading and fit helpers)
import sixvertex_ff_dimerization_scaling as ff              # noqa: E402  (exact K = 1 levels)

REPO = HERE.parents[1]
OUT_DATA = HERE
OUT_FIG = HERE
OUT_LOG = HERE
KS = (0.6, 0.75, 1.0, 1.25, 1.5, 1.75)
CHIS = (24, 32, 40, 48, 64)
CUTS = (0.03, 0.05, 0.075, 0.10)
N0S = (10, 12, 14)
L_FF = 4096
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def base_terms(K):
    t = [("const",), ("lin",)]
    if K < 0.75:
        t.append(rf.ex(2 ** (2 - 4 * K)))
    return t


def models(K):
    rp = 2 ** (2 - K)
    b = base_terms(K)
    grow = rf.ex(None, "end")
    quad_free = ("exp", lambda r: r ** 2, "end")
    quad_fixed = ("exp", lambda r, q=rp ** 2: q, "end")
    return {"G0": (b + [grow], None), "G1": (b + [rf.ex(rp, "end"), quad_fixed], rp), "G2": (b + [grow, quad_free], None)}


def amp(fr, which):
    """Coefficient of the growth (r^m) or quadratic (r^{2m}) term, both referred to n_e."""
    idx = [i for i, tm in enumerate(fr["terms"]) if tm[0] == "exp" and tm[2] == "end"]
    return fr["beta"][idx[0 if which == "A" else 1]]


# ---------------------------------------------------------------------------
def ff_X():
    """Exact X(s) - X(0) at K = 1 on a fine grid of s = delta L (L = L_FF), for interpolation in ln s."""
    x10, _ = ff.levels(L_FF, 0.0)
    s = np.exp(np.linspace(math.log(1e-4), math.log(3.0), 1500))
    X = np.array([-4 * (ff.levels(L_FF, v / L_FF)[0] - x10) for v in s])
    return s, X


def synthetic_rate(sg, Xg, N, cut, offsets=np.linspace(0.05, 0.95, 10)):
    """Fit F(s_n), s_n = s_c 2^{(n - n_c)/2}, over the N steps before |X| first exceeds cut, with the G0 protocol."""
    lnX = np.log(np.abs(Xg))
    s_c = math.exp(np.interp(math.log(cut), lnX, np.log(sg)))     # |F(s_c)| = cut
    out = []
    for u in offsets:
        ns = np.arange(1, N + 1, dtype=float)
        s_n = s_c * 2.0 ** ((ns - (N + u)) / 2)                       # step N + u would reach the cut
        X = -np.exp(np.interp(np.log(s_n), np.log(sg), lnX))
        fr = rf.fit(ns, X, [("const",), ("lin",), rf.ex(None, "end")])
        out.append(fr["r"] / 2.0)
    return float(np.mean(out)), float(np.min(out)), float(np.max(out))


# ---------------------------------------------------------------------------
def main():
    for d in (OUT_DATA, OUT_FIG, OUT_LOG):
        d.mkdir(exist_ok=True)
    log("# O(2) growth-rate deficit: g_d^4 term test; paper line numbers refer to main_1a.tex at e920332e")

    # ---- B: prediction at K = 1
    sg, Xg = ff_X()
    sel = sg <= 0.6
    c = ff.poly_coeffs(sg[sel], Xg[sel], 5)
    kappa_ff = c[1] / c[0] ** 2
    log(f"\n# B: free fermions (K = 1, L = {L_FF}): X = a s^2 + b s^4 + ..., a = {c[0]:.6f}, b = {c[1]:.6f}, "
        f"kappa = b/a^2 = {kappa_ff:.4f}")

    # ---- A: fits of the saved data
    rows = []
    log("\n# A: growth window n >= n0, |X| <= cut; r/2^(2-K) of G0 and G2, rms(G1)/rms(G0), kappa = B/A^2 of G1 and G2")
    for K in KS:
        rp = 2 ** (2 - K)
        for chi in CHIS:
            s = rf.series(K, chi)
            if s is None:
                continue
            ns, X, _, _ = s
            for n0 in N0S:
                for cut in CUTS:
                    gw = rf.growth_window(ns, X, n0, cut)
                    if gw is None:
                        continue
                    res = {}
                    for name, (terms, r) in models(K).items():
                        if gw.sum() < len(terms) + 4:
                            res = None
                            break
                        res[name] = rf.fit(ns[gw], X[gw], terms, r=r)
                    if not res:
                        continue
                    g0, g1, g2 = res["G0"], res["G1"], res["G2"]
                    A1, B1 = amp(g1, "A"), amp(g1, "B")
                    A2, B2 = amp(g2, "A"), amp(g2, "B")
                    row = dict(K=K, chi=chi, n0=n0, cut=cut, n_end=int(ns[gw][-1]), N=int(gw.sum()),
                               r0_over_pred=g0["r"] / rp, rms0=g0["rms"], A0=amp(g0, "A"),
                               rms1=g1["rms"], A1=A1, B1=B1, kappa1=B1 / A1 ** 2, kappa1_se=g1["se"][-1] / A1 ** 2,
                               r2_over_pred=g2["r"] / rp, rms2=g2["rms"], kappa2=B2 / A2 ** 2)
                    rows.append(row)
                    if n0 == 12 and cut == 0.10:
                        log(f"  K = {K}, chi = {chi} (n = 12..{row['n_end']}): G0 r/2^(2-K) = {row['r0_over_pred']:.4f}, "
                            f"rms {row['rms0']:.2e}; G1 rms {row['rms1']:.2e} (ratio {row['rms1'] / row['rms0']:.3f}), "
                            f"kappa = {row['kappa1']:+.3f} +- {row['kappa1_se']:.3f}; "
                            f"G2 r/2^(2-K) = {row['r2_over_pred']:.4f}, kappa = {row['kappa2']:+.3f}")

    # ---- spread of kappa over windows, per K and chi
    log("\n# kappa of G1 over windows (n0 = 10, 12, 14; cut = 0.03 ... 0.10): median [min, max]")
    for K in KS:
        for chi in CHIS:
            v = [r["kappa1"] for r in rows if r["K"] == K and r["chi"] == chi]
            if v:
                log(f"  K = {K}, chi = {chi}: {np.median(v):+.3f} [{min(v):+.3f}, {max(v):+.3f}] ({len(v)} windows)")

    # ---- B: synthetic rate at K = 1 for the windows of the data
    syn = []
    log("\n# B: synthetic X(n) = F(s_n) at K = 1, G0 protocol; window length N from the data at the same cut (n0 = 12)")
    for chi in CHIS:
        for cut in CUTS:
            d = [r for r in rows if r["K"] == 1.0 and r["chi"] == chi and r["n0"] == 12 and r["cut"] == cut]
            if not d:
                continue
            N = d[0]["N"]
            m, lo, hi = synthetic_rate(sg, Xg, N, cut)
            syn.append(dict(chi=chi, cut=cut, N=N, r_syn_over_pred=m, r_syn_lo=lo, r_syn_hi=hi,
                            r_data_over_pred=d[0]["r0_over_pred"]))
            log(f"  chi = {chi}, cut = {cut}: N = {N}, synthetic r/2 = {m:.4f} [{lo:.4f}, {hi:.4f}], data {d[0]['r0_over_pred']:.4f}")
    for cut in (0.01, 0.02):
        m, lo, hi = synthetic_rate(sg, Xg, 19, cut)
        log(f"  (synthetic only) cut = {cut}, N = 19: r/2 = {m:.4f} [{lo:.4f}, {hi:.4f}]")
        syn.append(dict(chi=0, cut=cut, N=19, r_syn_over_pred=m, r_syn_lo=lo, r_syn_hi=hi, r_data_over_pred=math.nan))

    def wcsv(path, rr):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0].keys()))
            w.writeheader()
            w.writerows(rr)

    wcsv(OUT_DATA / "sixvertex_o2_rate_deficit_fits.csv", rows)
    wcsv(OUT_DATA / "sixvertex_o2_rate_deficit_synthetic.csv", syn)
    (OUT_LOG / "sixvertex_o2_rate_deficit_report.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    plots(rows, syn, kappa_ff)


# ---------------------------------------------------------------------------
S = rf.S
INK, MUTED = rf.INK, rf.MUTED


def plots(rows, syn, kappa_ff):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.linewidth": 0.6})

    # (1) per K: rate of the paper fit, rms ratio G1/G0, kappa of G1 (n0 = 12, cut = 0.10; bars: range over windows)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for j, chi in enumerate(CHIS):
        main = sorted([r for r in rows if r["chi"] == chi and r["n0"] == 12 and r["cut"] == 0.10], key=lambda r: r["K"])
        if not main:
            continue
        Ks = np.array([r["K"] for r in main]) + 0.015 * (j - 2)
        axes[0].plot(Ks, [r["r0_over_pred"] for r in main], "o", ms=4.5, color=S[j], label=f"$\\chi={chi}$")
        axes[1].plot(Ks, [r["rms1"] / r["rms0"] for r in main], "o", ms=4.5, color=S[j])
        kap = np.array([r["kappa1"] for r in main])
        lo = np.array([min(x["kappa1"] for x in rows if x["K"] == r["K"] and x["chi"] == chi) for r in main])
        hi = np.array([max(x["kappa1"] for x in rows if x["K"] == r["K"] and x["chi"] == chi) for r in main])
        axes[2].errorbar(Ks, kap, yerr=[kap - lo, hi - kap], fmt="o", ms=4.5, color=S[j], elinewidth=0.8, capsize=0)
    axes[0].axhline(1, color=MUTED, lw=0.6)
    axes[0].set_ylabel("$r_{\\rm d}/2^{2-K}$ (paper fit G0)")
    axes[1].axhline(1, color=MUTED, lw=0.6)
    axes[1].set_ylabel("rms(G1) / rms(G0)")
    axes[2].axhline(0, color=MUTED, lw=0.6)
    axes[2].plot([1.07], [kappa_ff], "*", ms=12, color=INK, mec="white", mew=0.8, label="free fermions ($K=1$, drawn at 1.07)")
    axes[2].set_ylim(-6.5, 2.2)
    axes[2].set_ylabel("$\\kappa=B/A^2$ (G1, rate fixed)")
    axes[2].legend(fontsize=7, frameon=False, loc="upper right")
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    for ax in axes:
        ax.set_xlabel("$K$")
        ax.grid(alpha=0.15, lw=0.4)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "sixvertex_o2_rate_deficit_kappa.png", dpi=180)
    plt.close(fig)

    # (2) K = 1: r/2 of the paper fit against the window cut, data and free-fermion synthetic
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for j, chi in enumerate(CHIS):
        d = sorted([r for r in rows if r["K"] == 1.0 and r["chi"] == chi and r["n0"] == 12], key=lambda r: r["cut"])
        if d:
            ax.plot([r["cut"] for r in d], [r["r0_over_pred"] for r in d], "o-", ms=4, lw=0.8, color=S[j], label=f"data $\\chi={chi}$")
    sy = sorted([r for r in syn if r["chi"] in (0, 40)], key=lambda r: r["cut"])
    ax.plot([r["cut"] for r in sy], [r["r_syn_over_pred"] for r in sy], "-", lw=2, color=INK, label="free-fermion synthetic")
    ax.fill_between([r["cut"] for r in sy], [r["r_syn_lo"] for r in sy], [r["r_syn_hi"] for r in sy], color=INK, alpha=0.12, lw=0)
    ax.axhline(1, color=MUTED, lw=0.6)
    ax.set_xlabel("upper cut on $|X|$")
    ax.set_ylabel("$r_{\\rm d}/2^{2-K}$ at $K=1$ (G0)")
    ax.set_xlim(0, 0.105)
    ax.grid(alpha=0.15, lw=0.4)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "sixvertex_o2_rate_deficit_k1.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
