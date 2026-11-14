"""Task 1: dependence of r/2^(2-K) and the normalized amplitude ratio on the upper cut X_max, and
linear extrapolation X_max -> 0.

Usage: python3 task1_window.py   -> task1_cuts.csv, task1_extrap.csv, task1_summary.csv,
                                     fig_window_extrap.pdf/png  (printed tables as well)
Runs: O(2) main runs K = 0.75..1.75, chi = 24..64; B5 (chi = 96, K = 1, 1.5); B8 (K = 0.6, chi = 24..64).
Extrapolation: unweighted least squares y = y0 + b X_max over all usable cuts ("all"), and over cuts
X_max <= 0.03 ("small") and over 0.01 <= X_max <= 0.1 ("ge001") as stability checks.  Uncertainty: OLS standard error of the intercept from the
residual scatter (the cut windows are nested, so the points are not independent; the SE is indicative).
"""
import csv, math, os
import numpy as np
from common import HERE, series, fit_cut

CUTS = (0.1, 0.05, 0.03, 0.02, 0.01, 0.005)
RUNS = [(K, chi) for K in (0.75, 1.0, 1.25, 1.5, 1.75) for chi in (24, 32, 40, 48, 64)]
RUNS += [(1.0, 96), (1.5, 96)] + [(0.6, chi) for chi in (24, 32, 40, 48, 64)]


def linfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:          # no extrapolation from fewer than three cuts
        return float("nan"), float("nan"), float("nan"), len(x)
    A = np.vstack([np.ones_like(x), x]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    if len(x) > 2:
        s2 = ((A @ c - y) ** 2).sum() / (len(x) - 2)
        se = math.sqrt(s2 * np.linalg.inv(A.T @ A)[0, 0])
    else:
        se = float("nan")
    return c[0], se, c[1], len(x)


def main():
    rows, ext = [], []
    for K, chi in RUNS:
        X, XJ, _, _ = series(K, chi)
        per = []
        for xm in CUTS:
            f = fit_cut(K, X, XJ, xm)
            if f is None:
                continue
            f.update(K=K, chi=chi, X_max=xm, used=int("r" in f))
            rows.append(f)
            if "r" in f:
                per.append(f)
        e = dict(K=K, chi=chi, n_cuts=len(per), cuts_used=" ".join(f"{f['X_max']:g}" for f in per))
        std = next((f for f in per if f["X_max"] == 0.1), {})
        c02 = next((f for f in per if f["X_max"] == 0.02), {})
        for key in ("rate_ratio", "amp_ratio"):
            xs = [f["X_max"] for f in per]
            ys = [f[key] for f in per]
            y0, se, b, m = linfit(xs, ys)
            sm = [(x, y) for x, y in zip(xs, ys) if x <= 0.03]
            y0s, ses, _, ms = linfit([p[0] for p in sm], [p[1] for p in sm])
            lg = [(x, y) for x, y in zip(xs, ys) if x >= 0.01]
            y0l, sel, bl, ml = linfit([p[0] for p in lg], [p[1] for p in lg])
            fin = [y for y in ys if np.isfinite(y)]
            flags = []
            if m < 3:
                flags.append("fewer than 3 cuts")
            if np.isfinite(se) and se > 0.01:
                flags.append("SE>1%")
            if np.isfinite(y0s) and abs(y0s - y0) > max(2 * se if np.isfinite(se) else 0, 0.005):
                flags.append("all-cut vs small-cut extrapolations differ by %.3f" % (y0s - y0))
            if fin and (max(fin) - min(fin)) > 0.03:
                flags.append("spread over cuts %.3f" % (max(fin) - min(fin)))
            if any(f["r_at_grid_edge"] for f in per):
                flags.append("rate at grid edge")
            e.update({f"{key}_std": std.get(key, float("nan")), f"{key}_cut002": c02.get(key, float("nan")),
                      f"{key}_min_cut": ys[-1] if ys else float("nan"),
                      f"{key}_extrap": y0, f"{key}_extrap_se": se, f"{key}_slope": b,
                      f"{key}_extrap_ge001": y0l, f"{key}_extrap_ge001_se": sel,
                      f"{key}_extrap_smallcuts": y0s, f"{key}_extrap_smallcuts_se": ses,
                      f"{key}_flags": "; ".join(flags)})
        ext.append(e)
    write(os.path.join(HERE, "task1_cuts.csv"), rows,
          ["K", "chi", "X_max", "ne", "npts", "used", "umklapp", "r", "rate_ratio", "amp_ratio", "A", "r_at_grid_edge"])
    write(os.path.join(HERE, "task1_extrap.csv"), ext, list(ext[0].keys()))
    summ = summarize(ext)
    write(os.path.join(HERE, "task1_summary.csv"), summ, list(summ[0].keys()))
    report(rows, ext, summ)
    figure(rows, ext)


def write(path, rows, keys):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def summarize(ext):
    out = []
    for K in sorted({e["K"] for e in ext}):
        for grp, sel in (("chi24-64", lambda e: e["chi"] <= 64), ("chi32-96", lambda e: e["chi"] >= 32)):
            es = [e for e in ext if e["K"] == K and sel(e)]
            if not es:
                continue
            row = dict(K=K, chis=grp, n_runs=len(es))
            for key in ("rate_ratio", "amp_ratio"):
                for col in ("std", "cut002", "extrap", "extrap_ge001"):
                    v = np.array([e[f"{key}_{col}"] for e in es], float)
                    v = v[np.isfinite(v)]
                    row[f"{key}_{col}_median"] = float(np.median(v)) if len(v) else float("nan")
                    row[f"{key}_{col}_min"] = float(v.min()) if len(v) else float("nan")
                    row[f"{key}_{col}_max"] = float(v.max()) if len(v) else float("nan")
                se = np.array([e[f"{key}_extrap_se"] for e in es], float)
                row[f"{key}_extrap_se_median"] = float(np.nanmedian(se)) if np.isfinite(se).any() else float("nan")
                row[f"{key}_n_flagged"] = sum(bool(e[f"{key}_flags"]) for e in es)
            out.append(row)
    return out


def report(rows, ext, summ):
    print("Per run and cut: K chi X_max n_e npts rate_ratio amp_ratio")
    for r in rows:
        print(f"  {r['K']:.2f} {r['chi']:3d} {r['X_max']:<6g} {r['ne']:3d} {r['npts']:3d} "
              + (f"{r['rate_ratio']:.4f} {r['amp_ratio']:.4f}" if r["used"] else "(too few points)"))
    print("\nPer run: std(0.1) / cut 0.02 / extrap +- se  [small-cut extrap]  flags")
    for e in ext:
        for key in ("rate_ratio", "amp_ratio"):
            print(f"  K={e['K']:.2f} chi={e['chi']:3d} {key:10s} {e[key+'_std']:.4f} {e[key+'_cut002']:.4f} "
                  f"{e[key+'_extrap']:.4f}+-{e[key+'_extrap_se']:.4f} [>=0.01: {e[key+'_extrap_ge001']:.4f}+-{e[key+'_extrap_ge001_se']:.4f}; <=0.03: {e[key+'_extrap_smallcuts']:.4f}] "
                  f"ncuts={e['n_cuts']} {e[key+'_flags']}")
    print("\nSummary (median [min,max] over chi)")
    for s in summ:
        txt = f"  K={s['K']:.2f} {s['chis']:9s} n={s['n_runs']}"
        for key in ("rate_ratio", "amp_ratio"):
            for col in ("std", "cut002", "extrap", "extrap_ge001"):
                txt += f" | {key[:4]} {col} {s[f'{key}_{col}_median']:.3f} [{s[f'{key}_{col}_min']:.3f},{s[f'{key}_{col}_max']:.3f}]"
        print(txt)


def figure(rows, ext):
    """chi >= 32 only (chi = 24 fits at small cuts are unstable; see CSV).  Open symbols: individual cuts;
    filled symbols with error bars at X_max < 0: linear extrapolation over all cuts (OLS SE), offset by chi."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "mathtext.fontset": "cm",
                         "xtick.direction": "in", "ytick.direction": "in"})
    Ks = sorted({r["K"] for r in rows})
    col = dict(zip(Ks, ["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]))
    chis = (32, 40, 48, 64, 96)
    mk = {32: "^", 40: "o", 48: "v", 64: "D", 96: "*"}
    off = dict(zip(chis, (-0.010, -0.0085, -0.007, -0.0055, -0.004)))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for ax, key, lab in ((axes[0], "rate_ratio", r"$r_{\rm d}/2^{2-K}$"),
                         (axes[1], "amp_ratio", r"$(A_J/A)/[\alpha_J/(4K\alpha_{0,1})]$")):
        for e in ext:
            if e["chi"] not in chis:
                continue
            pts = [(r["X_max"], r[key]) for r in rows if r["K"] == e["K"] and r["chi"] == e["chi"] and r["used"]]
            if not pts:
                continue
            x, y = zip(*pts)
            c = col[e["K"]]
            ax.plot(x, y, "-", color=c, lw=0.5, alpha=0.5)
            ax.plot(x, y, mk[e["chi"]], color=c, ms=3.0, mfc="none", mew=0.7)
            if np.isfinite(e[key + "_extrap"]):
                ax.errorbar([off[e["chi"]]], [e[key + "_extrap"]], yerr=[e[key + "_extrap_se"]], fmt=mk[e["chi"]],
                            color=c, ms=3.0, lw=0.7, capsize=1.2)
        ax.axhline(1, color="k", lw=0.6)
        ax.axvline(-0.002, color="0.6", lw=0.5, ls="--")
        ax.set_xlim(-0.012, 0.104)
        ax.set_xlabel(r"upper cut $X_{\max}$  ($n_s=12$)")
        ax.set_ylabel(lab)
        ax.text(-0.007, 0.012, "extrap.", transform=ax.get_xaxis_transform(), ha="center", fontsize=6, color="0.4")
    axes[0].set_ylim(0.94, 1.06); axes[1].set_ylim(0.92, 1.12)
    hK = [Line2D([], [], color=col[K], lw=1.2, label=f"$K={K:g}$") for K in Ks]
    hc = [Line2D([], [], color="k", marker=mk[c], ls="none", mfc="none", ms=4, label=rf"$\chi={c}$") for c in chis]
    axes[0].legend(handles=hK, fontsize=6, frameon=False, ncol=3, loc="lower center")
    axes[1].legend(handles=hc, fontsize=6, frameon=False, ncol=3, loc="upper left")
    axes[0].set_title("(a) growth rate", fontsize=8)
    axes[1].set_title("(b) normalized amplitude ratio", fontsize=8)
    fig.tight_layout()
    for ext_ in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_window_extrap.{ext_}"), dpi=200)


if __name__ == "__main__":
    main()
