"""Paper 1b, Sec. II.C (fig:window): the seed protocol on one run, the 1b counterpart of Paper 1a Fig. 2(c).

|X(n)| of the lowest charged level for O(2), K=1.5, chi=40 (the chi-scan run), with the window n >= 12,
|X| <= 0.02 ending at n_end, the fit X = beta0 + beta1 n + A r_d^(n - n_end) with r_d = 2^(2-K) fixed,
the growth term extrapolated back to n = 12 (the seed), the breakdown |X| = 0.1 and n_e.

Check: the window (12..43) and g_d^2(12) (2.9989e-9) agree with
Otsuki/sixvertex/data/sixvertex_sg_xi_predict.csv (row "Yamazaki scan K1.5 chi40", cut 0.02, fixed rate); n_e = 48 is the last step with |X| <= 0.1
(the n_e column of that CSV is the first step beyond it, 49).

Usage: python3 fig_window_1b.py -> fig_window_1b.{pdf,png} here and research_note/paper/figures/fig_window_1b.pdf
"""
import math
import os
import shutil

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

import analyze as an

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_FIG = os.path.join(HERE, "..", "..", "..", "..", "research_note", "paper", "figures")
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "mathtext.fontset": "cm"})
INK, MUTED = "#0b0b0b", "#52514e"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#2e8b57"

# alpha_{0,1}(K) at K = 1.5 from Paper 1a App. B (same table as xi_sg_20261112/scripts/common.py)
import sys
sys.path.insert(0, os.path.join(HERE, "..", "..", "xi_sg_20261112", "scripts"))
import common  # noqa: E402


def main(K=1.5, chi=40, cut=0.02):
    d = os.path.join(HERE, "..", "data", f"CU1_K{K:.4f}_chi{chi}")
    lev, _ = an.load(d)
    X = {n: 1 - 4 * K * an.level(rows, "(1, 2)", 1) for n, rows in lev.items()}
    ns = np.array(sorted(X))
    xs = np.array([X[n] for n in ns])
    n_e = int([m for m in ns if m >= 12 and abs(X[m]) > 0.1][0]) - 1
    n_end = int([m for m in ns if m >= 12 and abs(X[m]) > cut][0]) - 1
    r = 2.0 ** (2 - K)
    nf = np.arange(12, n_end + 1)
    xf = np.array([X[int(m)] for m in nf])
    M = np.column_stack([np.ones_like(nf, float), nf.astype(float), r ** (nf - n_end)])
    (b0, b1, A), *_ = np.linalg.lstsq(M, xf, rcond=None)
    g2_12 = -A * r ** (12 - n_end) / (4 * K * common.alpha01(K))
    seed_X = abs(A) * r ** (12 - n_end)

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    keep = ns <= n_e + 3
    ax.axvspan(12, n_end, color="#eef3fb", lw=0, zorder=0)
    ax.semilogy(ns[keep], np.abs(xs[keep]), "o", ms=3.0, mfc=BLUE, mec="white", mew=0.5, zorder=3,
                label=r"$|X(n)|$")
    sub = np.array([X[int(m)] - b0 - b1 * m for m in nf])
    ax.semilogy(nf[sub * np.sign(A) > 0], np.abs(sub[sub * np.sign(A) > 0]), "s", ms=2.6, mfc="none",
                mec=GREEN, mew=0.6, zorder=3, label=r"$|X-\beta_0-\beta_1n|$ in window")
    ng = np.arange(12, n_e + 3.01, 0.25)
    ax.semilogy(ng, np.abs(A) * r ** (ng - n_end), "-", color=ORANGE, lw=1.3, zorder=2,
                label=r"$|A|\,r_{\rm d}^{\,n-n_{\rm end}}$, $r_{\rm d}=2^{2-K}$")
    ax.plot([12], [seed_X], "D", ms=4.5, mfc=ORANGE, mec="white", mew=0.6, zorder=4)
    ax.annotate("seed", (12, seed_X), xytext=(15, seed_X * 0.08), fontsize=7,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6), va="top")
    for y, lab in ((cut, r"$|X|=0.02$"), (0.1, r"$|X|=0.1$")):
        ax.axhline(y, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        ax.text(12.6, y * 1.25, lab, fontsize=6.5, color=MUTED)
    for x, lab in ((n_end, r"$n_{\rm end}$"), (n_e, r"$n_e$")):
        ax.vlines(x, 3e-6, 3, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        ax.text(x - 0.4, 1.2, lab, fontsize=6.5, color=MUTED, ha="right")
    ax.text(22, 1.2, "window", ha="center", fontsize=7, color=BLUE)
    ax.set_xlim(0, n_e + 3.5)
    ax.set_ylim(1e-9, 3)
    ax.set_xlabel(r"RG step $n$")
    ax.set_ylabel(r"$|X(n)|$")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_window_1b.{ext}"), dpi=300)
    shutil.copy(os.path.join(HERE, "fig_window_1b.pdf"), os.path.join(PAPER_FIG, "fig_window_1b.pdf"))
    print(dict(n_end=n_end, n_e=n_e, A=A, beta0=b0, beta1=b1, g2_12=g2_12, seed_X=seed_X))


if __name__ == "__main__":
    main()
