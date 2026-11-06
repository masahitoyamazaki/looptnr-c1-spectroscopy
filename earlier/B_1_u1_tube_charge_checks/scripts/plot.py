from pathlib import Path
import csv, json, math, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'
LEGACY_FITS=ROOT/'data/processed/odd_current_fits.csv'
def read_csv(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=240)
    plt.close(fig)

def plot_legacy_appendix() -> list[dict]:
    plt.rcParams.update({"font.family":"serif","mathtext.fontset":"dejavuserif","font.size":8.5,
                         "axes.labelsize":8.5,"xtick.labelsize":7.5,"ytick.labelsize":7.5,
                         "axes.linewidth":.7,"pdf.fonttype":42})
    rows = sorted(read_csv(LEGACY_FITS), key=lambda r: float(r["K_exact"]))
    k = [float(r["K_exact"]) for r in rows]
    tube = [float(r["tube_ratio"]) for r in rows]
    residual = [100*float(r["charge_linearity_residual"]) for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(3.42, 3.9), layout="constrained", sharex=True)
    for ax in axes:
        ax.set(xlim=(.43, 3.22))
        ax.grid(alpha=.16, linewidth=.5)
    axes[-1].set_xlabel(r"$K$")
    axes[0].plot(k, tube, "o", color="#b54d32", markersize=3.7)
    axes[0].axhline(2, color="#333333", linewidth=.9, linestyle="--")
    axes[0].set(ylabel=r"tube ratio $\rho$", ylim=(1.986, 2.006))
    axes[0].set_title("(a) tube ratio", fontsize=8.5, loc="left")
    axes[1].plot(k, residual, "o", color="#b54d32", markersize=3.7)
    axes[1].set(yscale="log", ylabel=r"charge residual $\epsilon_{\rm ch}$ (\%)", ylim=(.01, 50))
    axes[1].set_title(r"(b) $O_q\propto q$ check", fontsize=8.5, loc="left")
    save(fig, "u1_current_appendix_checks")
    return [dict(K=float(r["K_exact"]), vertex_c=float(r["vertex_c"]),
                 start=int(r["start"]), end=int(r["end"]), omega_fixed_deg=135.,
                 tube_ratio=float(r["tube_ratio"]), charge_linearity_residual=float(r["charge_linearity_residual"])) for r in rows]

