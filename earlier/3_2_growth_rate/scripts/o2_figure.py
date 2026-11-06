"""Figure 3 (fig:flows), refit: growth rate per RG step of the charged level against 2^(2-K).

Reads data/cft/o2_runs.csv and o2_levels.csv (full-precision levels exported from the saved CFT
records), fits every chi = 40 run with K < 2 with o2_fit.main_fit, and writes
  data/processed_o2/rates.csv, data/processed_o2/validation.json,
  figures/sixvertex_o2_rate.{pdf,png}
The results are compared with data/reference_o2/rates.csv.
"""
import csv
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import o2_fit as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed_o2"
REF = ROOT / "data/reference_o2"
FIG = ROOT / "figures"
PLAN_K = (0.6, 0.75, 1.0, 1.25, 1.5, 1.75)
CONTROL_K = 2.5
COLORS = {0.6: "#1b6ca8", 0.75: "#2a9d8f", 1.0: "#6a994e", 1.25: "#e9a23b", 1.5: "#d1495b", 1.75: "#8e5ea2"}
FIELDS = ["run", "chi", "K", "plan", "umklapp", "n_start", "n_end", "N", "r", "r_lo_1sigma", "r_hi_1sigma",
          "r_lo_all", "r_hi_all", "r_window_min", "r_window_max", "pred", "ratio", "ratio_lo", "ratio_hi", "rms"]


def plan_k(K):
    return next((k for k in PLAN_K if abs(K - k) < 1e-6), None)


def analyze(runs):
    rows = []
    for rid, run in sorted(runs.items(), key=lambda kv: kv[1]["K"]):
        if run["chi"] != 40 or run["K"] >= 2:
            continue
        ns, x, _ = F.series(run)
        m = F.main_fit(run["K"], ns, x)
        p = F.pred_rate(run["K"])
        rows.append(dict(run=rid, chi=run["chi"], K=run["K"], plan=plan_k(run["K"]) is not None,
                         umklapp=m["umklapp"], n_start=int(m["ns"][0]), n_end=int(m["ns"][-1]), N=m["N"],
                         r=m["r"], r_lo_1sigma=m["lo"], r_hi_1sigma=m["hi"], r_lo_all=m["lo_all"],
                         r_hi_all=m["hi_all"], r_window_min=m["spread"][0], r_window_max=m["spread"][1],
                         pred=p, ratio=m["r"] / p, ratio_lo=m["lo_all"] / p, ratio_hi=m["hi_all"] / p, rms=m["rms"]))
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()})


def compare(rows, ref_path, rtol=1e-8):
    """First difference from the reference table, or None."""
    with ref_path.open(newline="", encoding="utf-8") as f:
        ref = list(csv.DictReader(f))
    if len(ref) != len(rows):
        return f"{len(rows)} rows against {len(ref)} in the reference"
    for a, b in zip(rows, ref):
        for k in FIELDS:
            va, vb = a[k], b[k]
            if isinstance(va, float):
                vb = float(vb)
                if not (math.isclose(va, vb, rel_tol=rtol, abs_tol=1e-12) or (math.isnan(va) and math.isnan(vb))):
                    return f"run {a['run']} {k}: {va!r} against {vb!r}"
            elif str(va) != vb:
                return f"run {a['run']} {k}: {va} against {vb}"
    return None


def plot(rows):
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 9,
                         "axes.linewidth": 0.7, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.45, 2.6), layout="constrained")
    ax.axhline(1.0, color="#222222", lw=1.0)
    ax.axhspan(0.97, 1.03, color="#000000", alpha=0.06, lw=0)
    other = [r for r in rows if not r["plan"]]
    ax.scatter([r["K"] for r in other], [r["ratio"] for r in other], s=10, color="#b0b0b0", zorder=2,
               label=r"other $K$")
    for r in rows:
        k = plan_k(r["K"])
        if k is None:
            continue
        ax.errorbar([r["K"]], [r["ratio"]], yerr=[[r["ratio"] - r["ratio_lo"]], [r["ratio_hi"] - r["ratio"]]],
                    fmt="o", ms=4.5, color=COLORS[k], ecolor=COLORS[k], elinewidth=0.9, capsize=2, zorder=3,
                    label=rf"$K={k:g}$")
    ax.set(xlim=(0.5, 1.85), ylim=(0.9, 1.06), xlabel=r"$K$", ylabel=r"$r\,/\,2^{2-K}$")
    ax.grid(alpha=0.15, lw=0.5)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="lower right")
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"sixvertex_o2_rate.{ext}", dpi=220)
    plt.close(fig)


def main():
    if not F.verify_inputs(ROOT):
        print("o2 refit skipped: data/cft/o2_levels.csv is not in this CFT directory")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rows = analyze(F.load_runs(ROOT / "data/cft"))
    write_csv(OUT / "rates.csv", rows)
    plot(rows)
    ref = REF / "rates.csv"
    diff = compare(rows, ref) if ref.is_file() else "no reference"
    report = dict(passed=diff is None, first_difference=diff, reference="data/reference_o2/rates.csv",
                  reference_rtol=1e-8, rows=len(rows),
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   matplotlib=matplotlib.__version__))
    (OUT / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    for r in rows:
        if r["plan"]:
            print(f"K = {r['K']:.2f}  r = {r['r']:.3f}  r/2^(2-K) = {r['ratio']:.3f} ({r['ratio_lo']:.3f}-{r['ratio_hi']:.3f})"
                  f"  steps {r['n_start']}-{r['n_end']}  B {'yes' if r['umklapp'] else 'no'}")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
