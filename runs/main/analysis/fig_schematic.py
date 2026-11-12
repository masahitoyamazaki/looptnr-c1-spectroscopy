"""Schematic figure for main_en.tex (fig:schematic).

(a) one LoopTNR step: two-tensor unit cell, rescaling by sqrt2, rotation by 135 deg,
    and the original one-site translation as a fraction of a coarse lattice vector;
(b) readout tube geometry;
(c) |X(n)| of the lowest charged level for O(2), K=1.5, chi=40 with the regimes
    (transient, floor, growth, breakdown), the fit window and the seed.

Usage: python3 fig_schematic.py  ->  writes fig_schematic.{pdf,png} here and copies the
PDF to research_note/paper/figures/fig_schematic.pdf.
"""
import math
import os
import shutil

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle, Circle

import analyze as an

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_FIG = os.path.join(HERE, "..", "..", "..", "..", "research_note", "paper", "figures")
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "mathtext.fontset": "cm"})

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#c9c8c2"
BLUE, ORANGE = "#2a78d6", "#eb6834"      # validated categorical slots 1, 2


def lattice(ax, center, a, angle_deg, n=3, label_size=6.5, lw=0.8):
    """Square lattice of vertices with checkerboard T_A/T_B labels."""
    th = math.radians(angle_deg)
    e1 = np.array([math.cos(th), math.sin(th)]) * a
    e2 = np.array([-math.sin(th), math.cos(th)]) * a
    pts = {}
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            pts[i, j] = np.array(center) + i * e1 + j * e2
    for (i, j), p in pts.items():
        for di, dj in ((1, 0), (0, 1)):
            q = pts.get((i + di, j + dj))
            if q is not None:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=GRID, lw=lw, zorder=1)
    for (i, j), p in pts.items():
        if abs(i) == n or abs(j) == n:
            continue
        isA = (i + j) % 2 == 0
        ax.add_patch(Circle(p, 0.16 * a, facecolor=BLUE if isA else "white",
                            edgecolor=BLUE, lw=1.0, zorder=3))
    return e1, e2


def panel_a(ax):
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    e1, _ = lattice(ax, (2.0, 3.2), 0.75, 0, n=2)
    lattice(ax, (8.1, 3.2), 0.75 * math.sqrt(2), 135, n=2)
    ax.add_patch(FancyArrowPatch((3.85, 3.2), (4.95, 3.2), arrowstyle="-|>", mutation_scale=9,
                                 color=INK, lw=1.0))
    ax.text(4.4, 3.5, "step", ha="center", va="bottom", fontsize=7.5)
    ax.text(4.4, 2.9, r"$\times\sqrt{2}$" + "\n" + r"$135^\circ$", ha="center", va="top", fontsize=7)
    # original one-site translation vs coarse lattice vector
    p0 = np.array((2.0, 3.2))
    ax.add_patch(FancyArrowPatch(tuple(p0), tuple(p0 + e1), arrowstyle="-|>", mutation_scale=7,
                                 color=ORANGE, lw=1.3, zorder=4))
    ax.text(2.0 + 0.38, 3.2 + 0.2, r"$T_1$", color=INK, fontsize=7, ha="center", va="bottom")
    # direction of the same tensor leg before (x) and after (x') the step: rotated by 135 deg
    ax.add_patch(FancyArrowPatch((1.6, 5.35), (2.4, 5.35), arrowstyle="-|>", mutation_scale=7,
                                 color=INK, lw=1.0, zorder=4))
    ax.text(2.5, 5.35, r"$\hat x$", fontsize=7, ha="left", va="center")
    th = math.radians(135)
    q0 = np.array((10.55, 4.75))
    q1 = q0 + 0.8 * np.array((math.cos(th), math.sin(th)))
    ax.add_patch(FancyArrowPatch(tuple(q0), tuple(q1), arrowstyle="-|>", mutation_scale=7,
                                 color=INK, lw=1.0, zorder=4))
    ax.text(q1[0] - 0.05, q1[1] + 0.05, r"$\hat x'$", fontsize=7, ha="right", va="bottom")
    ax.text(2.0, 0.9, r"$T_A$ (filled)" + "\n" + r"$T_B$ (open)", ha="center", fontsize=6.5, color=MUTED)
    ax.text(8.1, 0.0, r"after $n$ steps, $T_1$ is a shift" + "\n" + r"by $2^{-n/2}$ of a lattice vector", ha="center", fontsize=6.5,
            color=MUTED)
    ax.text(0.05, 5.85, "(a)", fontsize=8.5, va="top")


def panel_b(ax):
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    x0, x1, yc, h = 0.8, 3.4, 3.0, 2.2
    ax.add_patch(Rectangle((x0, yc - h / 2), x1 - x0, h, facecolor="#eef3fb", edgecolor="none"))
    ax.plot([x0, x1], [yc + h / 2] * 2, color=INK, lw=0.9)
    ax.plot([x0, x1], [yc - h / 2] * 2, color=INK, lw=0.9)
    ax.add_patch(Ellipse((x0, yc), 0.8, h, facecolor="#eef3fb", edgecolor=INK, lw=0.9))
    ax.add_patch(Ellipse((x1, yc), 0.8, h, facecolor="none", edgecolor=INK, lw=0.9))
    for xx in np.linspace(x0 + 0.6, x1 - 0.6, 4):
        ax.add_patch(Ellipse((xx, yc), 0.8, h, facecolor="none", edgecolor=GRID, lw=0.6,
                             linestyle=(0, (2, 2))))
    ax.add_patch(FancyArrowPatch((x0, yc - h / 2 - 0.45), (x1, yc - h / 2 - 0.45),
                                 arrowstyle="<|-|>", mutation_scale=7, color=INK, lw=0.8))
    ax.text((x0 + x1) / 2, yc - h / 2 - 0.6, "transfer length\n" + r"$\sqrt{2}\,a_n$", ha="center",
            va="top", fontsize=7)
    ax.add_patch(FancyArrowPatch((x1 + 0.75, yc - h / 2), (x1 + 0.75, yc + h / 2),
                                 arrowstyle="<|-|>", mutation_scale=7, color=INK, lw=0.8))
    ax.text(x1 + 0.9, yc, "circum-\nference\n" + r"$2\sqrt{2}\,a_n$", ha="left", va="center",
            fontsize=7)
    ax.text((x0 + x1) / 2, yc + h / 2 + 0.35, r"transfer matrix: $\lambda_i\to x_i$",
            ha="center", va="bottom", fontsize=7.5)
    ax.text(0.05, 5.85, "(b)", fontsize=8.5, va="top")


def panel_c(ax):
    K, chi = 1.5, 40
    d = os.path.join(HERE, "..", "data", f"CU1_K{K:.4f}_chi{chi}")
    lev, _ = an.load(d)
    X = {n: 1 - 4 * K * an.level(rows, "(1, 2)", 1) for n, rows in lev.items()}
    ns = np.array(sorted(X))
    xs = np.array([X[n] for n in ns])
    out = [m for m in ns if m >= 12 and abs(X[m]) > 0.1]
    n_e = int(out[0]) - 1
    nf = np.arange(12, n_e + 1, dtype=float)
    _, lr, c, _ = an.fit_growth(nf, np.array([X[int(m)] for m in nf]), n_e, False, K)
    r = math.exp(lr)
    keep = ns <= n_e + 3
    ax.axvspan(12, n_e, color="#eef3fb", lw=0, zorder=0)
    ax.semilogy(ns[keep], np.abs(xs[keep]), "o", ms=3.2, mfc=BLUE, mec="white", mew=0.5,
                color=BLUE, zorder=3, label=r"$|X(n)|$")
    ng = np.arange(12, n_e + 3.01, 0.25)
    ax.semilogy(ng, np.abs(c[2]) * r ** (ng - n_e), "-", color=ORANGE, lw=1.4, zorder=2,
                label=r"growth term $\propto r^{\,n}$")
    seed = abs(c[2]) * r ** (12 - n_e)
    ax.plot([12], [seed], "D", ms=5, mfc=ORANGE, mec="white", mew=0.6, zorder=4)
    ax.annotate("seed", (12, seed), xytext=(15.5, seed * 12),
                fontsize=7, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6), va="bottom")
    ax.axhline(0.1, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.text(1.0, 0.13, r"$|X|=0.1$", fontsize=7, color=MUTED)
    ax.axvline(n_e, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.text(n_e + 0.5, 1.5e-3, r"$n_e$", fontsize=7, color=MUTED)
    ymax = 3
    ax.text(5, ymax / 3, "transient", ha="center", fontsize=7, color=MUTED)
    ax.text((12 + n_e) / 2 - 6, ymax / 3, "floor", ha="center", fontsize=7, color=MUTED)
    ax.text(n_e - 5, ymax / 3, "growth", ha="center", fontsize=7, color=MUTED)
    ax.text(27, 2.5e-2, "fit window", ha="center", fontsize=7, color=BLUE)
    ax.set_xlim(0, n_e + 3.5)
    ax.set_ylim(1e-7, ymax)
    ax.set_xlabel(r"RG step $n$")
    ax.set_ylabel(r"$|X(n)|$")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right", bbox_to_anchor=(0.93, 0.0))
    ax.set_title("(c)", loc="left", fontsize=8.5)
    return dict(n_e=n_e, r=r, seed_X=seed)


def main():
    fig = plt.figure(figsize=(7.0, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 0.95, 1.45], wspace=0.28)
    panel_a(fig.add_subplot(gs[0]))
    panel_b(fig.add_subplot(gs[1]))
    info = panel_c(fig.add_subplot(gs[2]))
    fig.subplots_adjust(left=0.0, right=0.99, bottom=0.16, top=0.93)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_schematic.{ext}"), dpi=300)
    os.makedirs(PAPER_FIG, exist_ok=True)
    shutil.copy(os.path.join(HERE, "fig_schematic.pdf"), os.path.join(PAPER_FIG, "fig_schematic.pdf"))
    print(info)


if __name__ == "__main__":
    main()
