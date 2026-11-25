"""Appendix figure for paper 1b: raw signed readout g_d^(pi)(n) on linear axes (companion of fig_signed).

Same data as panels (a), (b) of fig_signed.pdf (make_figures.py), but NOT divided by the first-order growth
(sqrt2)^{(2-K)n}.  K = 1.5, chi = 40 and 24, mu = 0, +-eps, +-eps/2, +-eps/4 (eps = 1e-3), n = 12..24.
The inset of each panel zooms on the uncompensated run (mu = 0) at n = 16..24 to show its sign.

Usage: <python> -I fig_signed_raw.py
Reads Fushio/docs/six_vertex/dimerization_injection/support/twist_campaign_20261109/twist.csv
"""
import csv, os
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
NMIN, NMAX = 12, 24


def series(sid):
    r = sorted((int(x["step"]), float(x["g"])) for x in tw if x["id"] == sid and x["g"] not in ("", None))
    return [(a, b) for a, b in r if NMIN <= a <= NMAX]


# same order, labels and colors as fig_signed (make_figures.py)
labels = [("plus0", r"$+\epsilon$"), ("plus1", r"$+\epsilon/2$"), ("plus2", r"$+\epsilon/4$"), ("zero", r"$0$"),
          ("minus2", r"$-\epsilon/4$"), ("minus1", r"$-\epsilon/2$"), ("minus0", r"$-\epsilon$")]
cmap = plt.get_cmap("coolwarm")

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
for k, (grp, title) in enumerate((("K1.5_chi40", r"(a) $K=1.5$, $\chi=40$"), ("K1.5_chi24", r"(b) $K=1.5$, $\chi=24$"))):
    for i, (suf, lab) in enumerate(labels):
        n, g = zip(*series(f"{grp}_{suf}"))
        col = "k" if suf == "zero" else cmap(1 - i / 6)
        ax[k].plot(n, [1e3 * b for b in g], "o-" if suf == "zero" else ".-",
                   color=col, lw=1.2 if suf == "zero" else 0.8, ms=3, label=lab, zorder=3 if suf == "zero" else 2)
    ax[k].axhline(0, color="0.6", lw=0.5)
    ax[k].set_xlabel("RG step $n$"); ax[k].set_title(title, fontsize=8.5)
    ax[k].set_xticks(range(NMIN, NMAX + 1, 4)); ax[k].set_xlim(NMIN - 0.6, NMAX + 0.6)

    # inset: mu = 0 only, n = 16..24, in units of 1e-4
    n0, g0 = zip(*[(a, b) for a, b in series(f"{grp}_zero") if a >= 16])
    ax[k].set_ylim(-21, 33)  # head room for the inset above the early steps
    ins = ax[k].inset_axes([0.09, 0.62, 0.36, 0.33])
    ins.plot(n0, [1e4 * b for b in g0], "o-", color="k", lw=1.0, ms=2.5)
    ins.axhline(0, color="0.6", lw=0.5)
    ins.set_xticks([16, 20, 24]); ins.tick_params(labelsize=6, length=2, pad=1.5)
    lo, hi = min(0.0, *[1e4 * b for b in g0]), max(0.0, *[1e4 * b for b in g0])
    pad = 0.15 * (hi - lo)
    ins.set_ylim(lo - pad, hi + pad)
    ins.text(0.04, 0.93 if k == 1 else 0.07, r"$\mu=0$: $10^4\,g_{\rm d}^{(\pi)}(n)$", transform=ins.transAxes,
             fontsize=6, va="top" if k == 1 else "bottom")
ax[0].set_ylabel(r"$10^3\,g_{\rm d}^{(\pi)}(n)$")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, [r"$\mu=$" + x for x in l], loc="lower center", ncol=7, fontsize=6.5, frameon=False)
fig.tight_layout(rect=(0, 0.06, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_signed_raw.{ext}"), dpi=200)
for grp in ("K1.5_chi40", "K1.5_chi24"):
    print(grp, "mu=0:", " ".join(f"{a}:{b:+.3e}" for a, b in series(f"{grp}_zero")),
          "| +eps:", " ".join(f"{a}:{b:+.3e}" for a, b in series(f"{grp}_plus0")))
