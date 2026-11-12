"""Fig.: measured and predicted xi against chi for the unperturbed runs (mu = 0).

Input (read only): TNR/Otsuki/sixvertex/data/sixvertex_sg_xi_predict.csv
Rows used: cut = 0.02, rate = fixed, mu = 0, settled = 1,
sources 'Yamazaki chi scan', 'Yamazaki B2', 'Yamazaki B5'.
Output: figures/fig_xi_vs_chi.pdf
"""
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "TNR", "Otsuki", "sixvertex", "data", "sixvertex_sg_xi_predict.csv")
rows = [r for r in csv.DictReader(open(SRC)) if r["cut"] == "0.02" and r["rate"] == "fixed"
        and r["source"] in ("Yamazaki chi scan", "Yamazaki B2", "Yamazaki B5")
        and float(r["mu"]) == 0 and r["settled"] == "1"]
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "mathtext.fontset": "cm"})
fig, ax = plt.subplots(figsize=(3.4, 2.8))
for K, col, mk in (("0.75", "C0", "o"), ("1.0", "C1", "^"), ("1.25", "C2", "s"), ("1.5", "C3", "D")):
    rs = sorted([r for r in rows if r["K"] == K], key=lambda r: int(r["chi"]))
    chi = [int(r["chi"]) for r in rs]
    ax.plot(chi, [float(r["xi_meas"]) for r in rs], mk, ms=3.5, color=col, label=rf"$K={float(K):g}$")
    ax.plot(chi, [float(r["xi_pred_end"]) for r in rs], "-", lw=0.6, color=col)
ax.set_yscale("log")
ax.set_xticks([16, 24, 32, 40, 48, 64, 96])
ax.set_xlabel(r"bond dimension $\chi$"); ax.set_ylabel(r"$\xi/a_0$")
ax.legend(fontsize=6.5, frameon=False, loc="lower right", ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "..", "figures", "fig_xi_vs_chi.pdf"))
print("rows", len(rows))
