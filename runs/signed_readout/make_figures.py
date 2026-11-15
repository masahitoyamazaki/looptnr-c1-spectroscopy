"""Figures for paper 3 v1 from the saved docs16 twist-campaign snapshot (no new RG runs).

Usage: ~/miniforge3/bin/python -I make_figures.py
Reads Fushio/docs/six_vertex/dimerization_injection/support/twist_campaign_20261109/{twist,observables,calibration}.csv
"""
import csv, json, math, os
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "..", "Fushio", "docs", "six_vertex", "dimerization_injection",
                   "support", "twist_campaign_20261109")
OUT = HERE
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "mathtext.fontset": "cm"})

tw = list(csv.DictReader(open(os.path.join(SRC, "twist.csv"))))
ob = list(csv.DictReader(open(os.path.join(SRC, "observables.csv"))))
cal = list(csv.DictReader(open(os.path.join(SRC, "calibration.csv"))))
proto = json.load(open(os.path.join(SRC, "protocol.json")))
EPS = {g["id"]: g["epsilon"] for g in proto["groups"]}

def series(rows, sid, key):
    r = sorted((int(x["step"]), float(x[key])) for x in rows if x["id"] == sid and x[key] not in ("", None))
    return [a for a, _ in r], [b for _, b in r]

def gamma(K):
    return math.sqrt(2) ** (2 - K)

# ---- Fig. 1: signed readout, rescaled by the first-order growth --------------------------
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.35))
labels = [("plus0", r"$+\epsilon$"), ("plus1", r"$+\epsilon/2$"), ("plus2", r"$+\epsilon/4$"), ("zero", r"$0$"),
          ("minus2", r"$-\epsilon/4$"), ("minus1", r"$-\epsilon/2$"), ("minus0", r"$-\epsilon$")]
cmap = plt.get_cmap("coolwarm")
for k, (grp, title) in enumerate((("K1.5_chi40", r"(a) $K=1.5$, $\chi=40$"), ("K1.5_chi24", r"(b) $K=1.5$, $\chi=24$"))):
    K = 1.5
    for i, (suf, lab) in enumerate(labels):
        n, g = series(tw, f"{grp}_{suf}", "g")
        n, g = zip(*[(a, b) for a, b in zip(n, g) if a <= 24])
        col = "k" if suf == "zero" else cmap(1 - i / 6)
        ax[k].plot(n, [b / gamma(K) ** a * 1e4 for a, b in zip(n, g)], "o-" if suf == "zero" else ".-",
                   color=col, lw=1.2 if suf == "zero" else 0.8, ms=3, label=lab)
    ax[k].axhline(0, color="0.6", lw=0.5)
    ax[k].set_xlabel("RG step $n$"); ax[k].set_title(title, fontsize=8.5)
ax[0].set_ylabel(r"$10^4\,g_{\rm d}^{(\pi)}(n)/r_{\rm d}^{\,n/2}$")
for grp, mk, lab in (("K1_chi40", "s", r"$K=1$, $\chi=40$"), ("K1.25_chi40", "^", r"$K=1.25$, $\chi=40$"),
                     ("K1.5_chi24", "o", r"$K=1.5$, $\chi=24$"), ("K1.5_chi32", "D", r"$K=1.5$, $\chi=32$"),
                     ("K1.5_chi40", "v", r"$K=1.5$, $\chi=40$")):
    eps = EPS[grp]
    rows = sorted((int(x["step"]), float(x["rescaled_slope"])) for x in cal
                  if x["group"] == grp and abs(float(x["epsilon"]) - eps / 4) < 1e-12 and int(x["step"]) >= 8)
    if not rows:
        continue
    ref = rows[-1][1]
    ax[2].plot([a for a, _ in rows], [b / ref for _, b in rows], mk + "-", ms=3, lw=0.8, label=lab)
ax[2].axhline(1, color="0.6", lw=0.5)
ax[2].set_xlabel("RG step $n$"); ax[2].set_ylabel(r"$s(n)/r_{\rm d}^{\,n/2}$ (normalized at $n_{\rm cal}$)")
ax[2].set_title(r"(c) linear response, $\epsilon/4$", fontsize=8.5); ax[2].legend(fontsize=5.3, frameon=False)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, [r"$\mu=$"+x for x in l], loc="lower center", ncol=7, fontsize=6.5, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_signed.{ext}"), dpi=200)

# ---- Fig. 2: frozen prediction, charged-level deviation ----------------------------------
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.35), sharey=True)
for k, (grp, title) in enumerate((("K1_chi40", r"(a) $K=1$, $\chi=40$"), ("K1.5_chi40", r"(b) $K=1.5$, $\chi=40$"),
                                  ("K1.5_chi24", r"(c) $K=1.5$, $\chi=24$"))):
    for suf, sty, lab in (("zero", "k.-", r"$\mu=0$"), ("predicted", "C3o-", r"$\mu_{\rm pred}$"),
                          ("sensitivity0.9", "C0.--", r"$0.9\,\mu_{\rm pred}$"), ("sensitivity1.1", "C2.--", r"$1.1\,\mu_{\rm pred}$")):
        n, X = series(ob, f"{grp}_{suf}", "X")
        if not n:
            continue
        ax[k].semilogy(n, [abs(x) for x in X], sty, ms=2.2, lw=0.8, label=lab)
    ax[k].axhline(0.1, color="0.5", lw=0.6, ls=":")
    ax[k].set_xlabel("RG step $n$"); ax[k].set_title(title, fontsize=8.5)
ax[0].set_ylabel(r"$|X(n)|=|1-4Kx_{0,1}(n)|$"); ax[0].set_ylim(1e-6, 0.5)
ax[1].legend(fontsize=6, frameon=False, loc="lower right")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_compensation.{ext}"), dpi=200)
print("ok")
