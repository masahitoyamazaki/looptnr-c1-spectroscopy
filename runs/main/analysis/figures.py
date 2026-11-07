"""Figures for main_en.tex from the single-environment baseline (chi=96 rows of o2_fits.csv: runs B5 in runs/additional).
Usage: python3 figures.py DATA_DIR_BATCH2   (needs o2_fits.csv from analyze.py)"""
import csv, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts (journals reject Type 3)
import matplotlib.pyplot as plt
from analyze import load, level
HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm"})

# (1) seed coupling versus chi
rows = [r for r in csv.DictReader(open(os.path.join(HERE, "o2_fits.csv"))) if r.get("gd2_n12")]
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for K, mk in zip(("0.75", "1.0", "1.25", "1.5", "1.75"), "osD^v"):
    pts = sorted((int(r["chi"]), float(r["gd2_n12"])) for r in rows if float(r["K"]) == float(K))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", ms=4, lw=0.8, label=f"$K={float(K):g}$")
chi = np.array([16, 96]); ax.plot(chi, 1e-6 * (chi / 16.0) ** -6, "k--", lw=0.8, label=r"$\propto\chi^{-6}$ (guide)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([16, 24, 32, 40, 48, 64, 96]); ax.set_xticklabels([16, 24, 32, 40, 48, 64, 96])
ax.minorticks_off(); ax.set_xlabel(r"$\chi$"); ax.set_ylabel(r"$g_{\rm d}^2$ at $n=12$")
ax.legend(fontsize=7, ncol=2, frameon=False); fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_seed_vs_chi.pdf")); fig.savefig(os.path.join(HERE, "fig_seed_vs_chi.png"), dpi=200)

# (2) eigenvalue phase of the neutral e=+-1 pair versus the prediction from Re g_J
base = sys.argv[1]
fig, ax = plt.subplots(figsize=(3.4, 2.8))
u1 = {(float(r["K"]), int(r["chi"])): r for r in csv.DictReader(open(os.path.join(HERE, "u1_fits.csv")))}
for K, c in zip((1.25, 1.5, 1.75), ("C0", "C1", "C2")):
    r = u1[(K, 40)]
    d = os.path.join(base, f"U1_K{K:.4f}_chi40"); lev, steps = load(d)
    n_e = int(r["n_e"]); g = float(r["gamma"]); w = math.radians(float(r["omega_deg"]))
    # refit u, v at fixed gamma, omega on the same 10-point window
    ns = list(range(n_e - 9, n_e + 1)); t = np.array([m - n_e for m in ns], float)
    o = np.array([(level(lev[m], "U₁](1)", 1) - level(lev[m], "U₁](-1)", 1)) / 2 for m in ns])
    M = np.vstack([g ** t * np.cos(w * t), g ** t * np.sin(w * t)]).T
    (u, v), *_ = np.linalg.lstsq(M, o, rcond=None)
    tt = np.arange(-12, 0.01, 1.0)
    pred = 2 * math.pi * K * g ** tt * np.abs(u * np.sin(w * tt) - v * np.cos(w * tt))
    meas = [np.mean(sorted((abs(float(x["phase"])) for x in lev[n_e + int(s)] if "U₁](0)" in x["sector"]
                            and abs(float(x["x"]) - K) < 0.1), reverse=True)[:2]) for s in tt]
    ax.plot(n_e + tt, pred, "-", color=c, lw=0.8)
    ax.plot(n_e + tt, meas, "o", color=c, ms=3.5, mfc="none", label=f"$K={K:g}$")
ax.set_yscale("log"); ax.set_xlabel("RG step $n$"); ax.set_ylabel(r"$|\arg(\lambda_{\pm1,0}/\lambda_0)|$")
ax.legend(fontsize=7, frameon=False); fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_u1_phase.pdf")); fig.savefig(os.path.join(HERE, "fig_u1_phase.png"), dpi=200)
print("ok")
