"""Figure for paper 1b, Sec. VI: corrections during the RG with four constraints (P3_4c, every two steps).

(a), (b) |X(n)| of the lowest charged level at K = 1.5 and 1, uncompensated main runs (dashed) and corrected
runs (solid), chi = 24, 32, 40.  (c) BKT point: 4/g_t from the triplet, corrected (solid) and uncompensated
(dashed), with the two-loop flow started from the chi = 40 value at step 12 (black).

Usage: python3 -I make_figure.py   (writes fig_midrg.pdf/.png next to this file)
Inputs: out/*.toml of this folder and the stored spectra of the uncompensated main runs
        (../main/data in the TNR layout, ../main/data in the public repository).
"""
import csv, gzip, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.5))
for a, K in zip(ax[:2], (1.5, 1.0)):
    for chi in CHIS:
        n, X = main_X(K, chi)
        a.plot(n[n >= 4], X[n >= 4], "--", color=COL[chi], lw=0.9)
        S = run(K, chi, "P3_4c")
        a.plot([s["step"] for s in S], [abs(s["X"]) for s in S], "-", color=COL[chi], lw=0.9, label=rf"$\chi={chi}$")
    a.axhline(0.1, color="0.5", lw=0.6, ls=":")
    a.set(yscale="log", ylim=(1e-5, 0.5), xlim=(0, 165), xlabel="RG step $n$", title=f"({'ab'[K == 1.0]}) $K={K:g}$")
ax[0].set_ylabel(r"$|X(n)|$")
ax[0].legend(fontsize=7, frameon=False, loc="lower right")

a = ax[2]
for chi in CHIS:
    for d, ls in (("P3_4c", "-"), ("none", "--")):
        S = run(0.5, chi, d)
        n = np.array([s["step"] for s in S]); x01 = np.array([s["x01"] for s in S]); em = np.array([s["eminus"] for s in S])
        g = 4 * (0.5 - (2 * x01 + em) / 3)
        m = (n >= 8) & (g > 0) & (g < 0.4)
        if d == "none":  # before the dimerization grows (n_e = 19, 28, 26; cf. paper 1a, Sec. IV E)
            m &= n <= {24: 16, 32: 22, 40: 22}[chi]
        a.plot(n[m], 4 / g[m], ls, color=COL[chi], lw=0.9)
S = run(0.5, 40, "P3_4c"); g = next(4 * (0.5 - (2 * s["x01"] + s["eminus"]) / 3) for s in S if s["step"] == 12)
ns, gs = [12], [g]
for k in range(12, 160):
    for _ in range(100):  # integrate dg/dl = -g^2 - g^3/2 over one step, dl = ln sqrt2
        g += -(g * g + 0.5 * g ** 3) * math.log(math.sqrt(2)) / 100
    ns.append(k + 1); gs.append(g)
a.plot(ns, 4 / np.array(gs), "k-", lw=0.6, label="two-loop flow")
a.set(xlabel="RG step $n$", ylabel=r"$4/\hat g$", title="(c) $K=1/2$", xlim=(0, 165), ylim=(0, 260))
a.legend(fontsize=7, frameon=False, loc="upper left")
fig.tight_layout(w_pad=0.6)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"fig_midrg.{ext}"), dpi=200)
print("wrote fig_midrg")
