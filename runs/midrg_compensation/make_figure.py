"""Figure for paper 1b, Sec. VI: corrections during the RG with four constraints (P3_4c, every two steps).

(a), (b) K = 1.5 and 1, chi = 24, 32, 40: charge-ratio error max(|x02/4x01 - 1|, |x03/9x01 - 1|) (top) and
K_n - K (bottom), corrected (solid) and uncompensated main runs (dashed).  (c) BKT point: 4/g_t from the triplet, corrected (solid) and uncompensated
(open circles), with the two-loop flow started from the chi = 40 value at step 12 (black).

Usage: python3 -I make_figure.py   (writes fig_midrg.pdf/.png next to this file)
Inputs: out/*.toml of this folder and the stored spectra of the uncompensated main runs
        (../main/data in the TNR layout, ../main/data in the public repository).
"""
import csv, gzip, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "mathtext.fontset": "cm",  # common style of the paper figures
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
                     "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False})
TICKS = [0, 40, 80, 120, 160]
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = next(p for p in (os.path.join(HERE, "..", "main", "data"), os.path.join(HERE, "..", "main", "data"))
            if os.path.isdir(p))
CHIS, COL = (24, 32, 40), {24: "#4C72B0", 32: "#DD8452", 40: "#55A868"}


def run(K, chi, d):
    with open(os.path.join(HERE, "out", f"K{K:.2f}_chi{chi}_{d}_int2.toml"), "rb") as f:
        return tomllib.load(f)["steps"]


def main_X(K, chi):
    rows = csv.DictReader(gzip.open(os.path.join(MAIN, f"CU1_K{K:.4f}_chi{chi}", "levels.csv.gz"), "rt"))
    x = {}
    for r in rows:
        if "(1, 2)" in r["sector"] and r["rank"] == "1":
            x[int(r["step"])] = float(r["x"])
    n = np.array(sorted(x)); return n, np.abs(1 - 4 * K * np.array([x[k] for k in n]))


# (a), (b): adapted from Fushio's proposal for 1b R-16 (research_note/paper/reviews/Fushio/figures/
# paper1b_fig9ab_running_k.py): charge-ratio error (top) and K_n - K (bottom), corrected (solid) and
# uncompensated (dashed; K_n only over the floor window of paper 1a, ending at the open circle).
ANALYSIS = os.path.join(MAIN, "..", "analysis")


def charge_error(x1, x2, x3):
    return 100 * max(abs(x2 / (4 * x1) - 1), abs(x3 / (9 * x1) - 1))


def uncompensated(k0, chi, floor_end):
    by = {}
    with gzip.open(os.path.join(MAIN, f"CU1_K{k0:.4f}_chi{chi}", "levels.csv.gz"), "rt") as f:
        for r in csv.DictReader(f):
            n = int(r["step"])
            if n >= 12:
                by.setdefault(n, {}).setdefault(r["sector"].split("]")[-1], []).append(float(r["x"]))
    drift, errors = [], []
    for n, sec in sorted(by.items()):
        x1, x2, x3 = (min(sec[f"({q}, 2)"]) for q in (1, 2, 3))
        errors.append((n, charge_error(x1, x2, x3)))
        if n > floor_end:
            continue
        even = sorted(sec["(0, 0)"])[1]
        odd = sorted(sec["(0, 1)"])[:3]
        if k0 == 1:  # identify the two currents as in run.jl (closest pair)
            pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
            scalar = min(pairs, key=lambda p: abs(odd[p[0]] - odd[p[1]]))[2]
        else:
            scalar = 2
        drift.append((n, 0.5 * math.sqrt((even + odd[scalar]) / (2 * x1)) - k0))
    return np.array(drift).T, np.array(errors).T


with open(os.path.join(ANALYSIS, "drift_summary.csv")) as f:
    WINDOWS = {(float(r["K"]), int(r["chi"])): int(r["n_last"]) for r in csv.DictReader(f) if r["n_last"]}

fig = plt.figure(figsize=(7.0, 3.5))
gs = fig.add_gridspec(2, 3, width_ratios=(1, 1, 1.25), left=0.095, right=0.985, top=0.93, bottom=0.13,
                      wspace=0.36, hspace=0.1)
for col, K in enumerate((1.5, 1.0)):
    top = fig.add_subplot(gs[0, col]); bot = fig.add_subplot(gs[1, col], sharex=top)
    for chi in CHIS:
        S = [s for s in run(K, chi, "P3_4c") if s["step"] >= 12]
        ku, eu = uncompensated(K, chi, WINDOWS[(K, chi)])
        top.plot([s["step"] for s in S], [charge_error(s["x01"], s["x02"], s["x03"]) for s in S], "-", color=COL[chi], lw=0.8,
                 label=rf"$\chi={chi}$")
        top.plot(*eu, "--", color=COL[chi], lw=0.8)
        bot.plot([s["step"] for s in S], [s["Kn"] - K for s in S], "-", color=COL[chi], lw=0.8)
        bot.plot(*ku, "--", color=COL[chi], lw=0.8)
        bot.plot(*ku[:, -1], "o", ms=2.8, mew=0.65, mec=COL[chi], mfc="white", zorder=4)
    top.axhline(1, color="0.6", lw=0.6, ls=":")
    bot.axhline(0, color="0.6", lw=0.55)
    top.set(yscale="log", ylim=(0.002, 115), title=f"({'ab'[K == 1.0]}) $K={K:g}$")
    top.set_yticks([0.01, 0.1, 1, 10, 100]); top.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
    top.tick_params(labelbottom=False)
    bot.set(ylim=(-0.053, 0.016), yticks=[-0.04, -0.02, 0, 0.01], xlim=(8, 166), xticks=[12, 40, 80, 120, 160],
            xlabel="RG step $n$")
    if col == 0:
        top.set_ylabel("charge-ratio error (%)")
        bot.set_ylabel(r"$K_n-K$")
        top.legend(fontsize=6.5, loc="upper right", ncol=1, handlelength=1.5)
    else:
        top.legend(handles=[matplotlib.lines.Line2D([], [], color="0.3", ls="-", lw=0.8, label="corrected"),
                            matplotlib.lines.Line2D([], [], color="0.3", ls="--", lw=0.8, label="uncompensated")],
                   fontsize=6.5, loc="upper right", handlelength=1.8)
axc = fig.add_subplot(gs[:, 2])
a = axc
for chi in CHIS:
    for d, ls in (("P3_4c", "-"), ("none", "o")):
        S = run(0.5, chi, d)
        n = np.array([s["step"] for s in S]); x01 = np.array([s["x01"] for s in S]); em = np.array([s["eminus"] for s in S])
        g = 4 * (0.5 - (2 * x01 + em) / 3)
        m = (n >= 8) & (g > 0) & (g < 0.4)
        if d == "none":  # before the dimerization grows (n_e = 19, 28, 26; cf. paper 1a, Sec. IV E)
            m &= n <= {24: 16, 32: 22, 40: 22}[chi]
        if d == "none":  # open circles, so that the short uncompensated segment is visible on top of the solid line
            a.plot(n[m], 4 / g[m], ls, mfc="white", mec=COL[chi], ms=3, mew=0.8, zorder=3)
        else:
            a.plot(n[m], 4 / g[m], ls, color=COL[chi], lw=0.9)
ins = a.inset_axes([0.14, 0.52, 0.32, 0.29])  # zoom on the steps where the uncompensated runs are still usable
for chi in CHIS:
    for d in ("P3_4c", "none"):
        S = run(0.5, chi, d)
        n = np.array([s["step"] for s in S]); x01 = np.array([s["x01"] for s in S]); em = np.array([s["eminus"] for s in S])
        g = 4 * (0.5 - (2 * x01 + em) / 3)
        m = (n >= 8) & (n <= 24) & (g > 0) & (g < 0.4)
        if d == "none":
            m &= n <= {24: 16, 32: 22, 40: 22}[chi]
            ins.plot(n[m], 4 / g[m], "o", mfc="white", mec=COL[chi], ms=2.5, mew=0.7, zorder=3)
        else:
            ins.plot(n[m], 4 / g[m], "-", color=COL[chi], lw=0.8)
ins.set_xlim(7, 25); ins.set_xticks([8, 16, 24]); ins.tick_params(labelsize=6.5, length=2)
S = run(0.5, 40, "P3_4c"); g = next(4 * (0.5 - (2 * s["x01"] + s["eminus"]) / 3) for s in S if s["step"] == 12)
ns, gs = [12], [g]
for k in range(12, 160):
    for _ in range(100):  # integrate dg/dl = -g^2 - g^3/2 over one step, dl = ln sqrt2
        g += -(g * g + 0.5 * g ** 3) * math.log(math.sqrt(2)) / 100
    ns.append(k + 1); gs.append(g)
a.plot(ns, 4 / np.array(gs), "k-", lw=0.6, label="two-loop flow")
a.plot([], [], "o", mfc="white", mec="0.3", ms=3, mew=0.8, label="uncompensated")
a.set(xlabel="RG step $n$", ylabel=r"$4/\hat g$", title="(c) $K=1/2$", xlim=(0, 165), xticks=TICKS, ylim=(0, 260))
a.legend(fontsize=7, frameon=False, loc="lower right")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"fig_midrg.{ext}"), dpi=200)
print("wrote fig_midrg")
