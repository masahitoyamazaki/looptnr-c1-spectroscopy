"""Shared inputs and conventions for the paper-2 figures and numbers (no tensor computation).

Inputs (read only):
  TNR/Otsuki/sixvertex/data/sixvertex_sg_xi_predict.csv        one row per run, window cut and rate choice
  TNR/Otsuki/sixvertex/data/sixvertex_sg_xi_predict_steps.csv  steps used in each fit
  TNR/Otsuki/sixvertex/data/sixvertex_sg_xi_levels.csv         levels per step of each run
  TNR/Otsuki/sixvertex/data/sixvertex_g2_alpha.csv             alpha_{0,1}(K) of paper 1
  TNR/Yamamoto/finite_bonddim/data/report10_correlation_20261109/{comparison,profiles}.csv

Convention of this paper (the "protocol"): cut |X| <= 0.02, rate fixed to 2^(2-K), coupling read at the end of the
window, xi measured as the average of the last two steps before the numerical floor.  A run is counted once per
distinct trajectory: rows of two sources with identical measured xi are the same stored run and are merged.
"""
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
TNR = HERE.parents[3]
OTS = TNR / "Otsuki" / "sixvertex" / "data"
YAM = TNR / "Yamamoto" / "finite_bonddim" / "data" / "report10_correlation_20261109"
FIG = PAPER / "figures"

KS = (0.75, 1.0, 1.25, 1.5, 1.75)
KCOL = {0.75: "C0", 1.0: "C1", 1.25: "C2", 1.5: "C3", 1.75: "C4"}
KMARK = {0.75: "o", 1.0: "^", 1.25: "s", 1.5: "D", 1.75: "v"}
CUT = 0.02


def style():
    plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "mathtext.fontset": "cm",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
                         "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
                         "lines.markersize": 3.5})


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [{k: num(v) for k, v in r.items()} for r in rows]


def kappa(p):
    return (math.gamma(p / (p + 1)) / math.gamma(1 / (p + 1)) / math.pi
            * (math.sqrt(math.pi) * math.gamma((p + 1) / 2) / (2 * math.gamma(p / 2))) ** (2 / (p + 1)))


def kappa_K(K):
    return kappa(K / (2 - K))


def ell(n):
    return 2 * math.sqrt(2) * math.sqrt(2) ** n


def xi_from_g(K, n, g):
    return ell(n) / (2 * math.pi) * (math.sqrt(2) * kappa_K(K) / abs(g)) ** (1 / (2 - K))


def alpha01(K):
    return next(r["m1"] for r in read(OTS / "sixvertex_g2_alpha.csv") if abs(r["K"] - K) < 1e-9)


def ok(r):
    v = r.get("ratio_end")
    return isinstance(v, float) and v == v


def runs(rows, K, cut=CUT, rate="fixed"):
    """Settled runs at K (gapped at the last used step), saved checkpoints excluded, duplicates merged."""
    sel = [r for r in rows if r["K"] == K and r["cut"] == cut and r["rate"] == rate and r["settled"] == 1
           and r["source"] != "saved" and ok(r)]
    out, seen = [], set()
    for r in sel:
        key = (r["chi"], round(r["xi_meas"], 3))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def predict_rows():
    return read(OTS / "sixvertex_sg_xi_predict.csv")
