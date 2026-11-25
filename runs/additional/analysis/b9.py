"""Analysis of runs B9: staggered a/b weights (STAGGER_AB = h), the deformation with the quantum numbers of V_d.

Usage:  python3 b9.py OUT_DIR      (B9_K*_chi*_ab*/ with levels.csv, steps.csv from Miyabi-C)

For each run g_d^2(n=12) and n_e are obtained as in b1_b2.py (fit of X = 1 - 4K x_{0,1} from n = 12 to the
step before |X| > 0.1).  If the response is linear, g_d(12; h) = g_seed + c h, so g_d^2(12) is a parabola whose
minimum at h* = -g_seed/c cancels the truncation seed at n = 12.  Spectra fix only g_d^2, so (g_seed, c) is
determined up to a common sign; we report c > 0, i.e. the sign of g_seed relative to the direction of h.
Prints the fit and n_e(h), writes b9.csv and fig_b9.pdf/png.

2026-11-25: the parabola is fitted with a global search (global_fit, copied from
seed_refit_20261112/refit_b9.py); the earlier fit from four starting values landed in local minima for
(K,chi)=(0.75,32), (1,40), (1.25,32), (1.5,32), (1.75,32).  The raw run directories are on Miyabi-C;
`python3 b9.py --from-csv` refits the per-run g2 and n_e stored in b9.csv (old outputs: old_20261125_localmin/).
"""
import csv, glob, math, os, sys
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from b1_b2 import seed   # noqa: E402


def global_fit(mu, g2):
    """Global least squares of ln g2 = 2 ln|g_seed + rho*mu| (rho>0).
    2026-11-25 fix: the earlier version ran least_squares from 8 starting values only and landed in local
    minima for (K,chi)=(1.75,32), (1.25,40) and in several jackknife samples (see
    Yamazaki/runs/sec4_check_20261125/local_minima.txt).  Now: with g2 = rho^2 (mu-mu*)^2 the optimal ln rho at
    fixed mu* is the mean residual, so the cost is profiled over mu* on a dense grid (|mu*| <= 3 max|mu|),
    the 40 best grid minima are refined by bounded 1D minimization, and the best one is polished with
    least_squares in (g_seed, rho) (same residual and bounds as before; gives the Jacobian for the errors)."""
    lg = np.log(g2)

    def prof(ms):
        d = lg - 2 * np.log(np.abs(mu - ms) + 1e-300)
        c = d.mean()
        return float(np.sum((d - c) ** 2)), c

    span = 3 * np.max(np.abs(mu)) + 1e-6
    grid = np.linspace(-span, span, 600001)
    h = grid[1] - grid[0]
    Cc = []
    for chunk in np.array_split(grid, 60):
        D = lg[None, :] - 2 * np.log(np.abs(mu[None, :] - chunk[:, None]) + 1e-300)
        Cc.append(np.sum((D - D.mean(1, keepdims=True)) ** 2, 1))
    Cc = np.nan_to_num(np.concatenate(Cc), nan=np.inf)
    loc = np.flatnonzero((Cc[1:-1] <= Cc[:-2]) & (Cc[1:-1] <= Cc[2:])) + 1
    cand = sorted(zip(Cc[loc], grid[loc]))
    best = None
    for _, m0 in cand[:40]:
        r = minimize_scalar(lambda m: prof(m)[0], bounds=(m0 - h, m0 + h), method="bounded", options={"xatol": 1e-17})
        if best is None or r.fun < best.fun:
            best = r
    rho = math.exp(prof(best.x)[1] / 2)
    f = lambda p: 2 * np.log(np.abs(p[0] + p[1] * mu) + 1e-30) - np.log(g2)
    res = least_squares(f, [-rho * best.x, rho], bounds=([-1, 0], [1, 1e4]), x_scale=[abs(rho * best.x) + 1e-12, rho])
    assert 2 * res.cost <= best.fun * (1 + 1e-6) + 1e-12, (2 * res.cost, best.fun)
    return res


def parse(name):
    K = float(name.split("_K")[1].split("_")[0]); chi = int(name.split("_chi")[1].split("_")[0])
    t = name.split("_ab")[1]
    return K, chi, (0.0 if t == "0" else float(t))


def main(out):
    runs = {}
    if out == "--from-csv":
        for r in csv.DictReader(open(os.path.join(HERE, "old_20261125_localmin", "b9.csv"))):
            K, chi, h = float(r["K"]), int(r["chi"]), float(r["h"])
            runs.setdefault((K, chi), []).append((h, dict(g2=float(r["g2_n12"]), ne=int(r["n_e"]),
                                                           r=float(r["r_over_pred"]) * 2 ** (2 - K))))
    else:
        for d in sorted(glob.glob(os.path.join(out, "B9_K*_chi*_ab*"))):
            K, chi, h = parse(os.path.basename(d))
            f = seed(d, K)
            runs.setdefault((K, chi), []).append((h, f))
    rows, fits = [], {}
    for (K, chi), pts in sorted(runs.items()):
        pts = sorted(pts)
        ok = [(h, f) for h, f in pts if f]
        if len(ok) < 3:
            print(f"K={K} chi={chi}: only {len(ok)} finished runs, skipped")
            continue
        hs = np.array([h for h, _ in ok]); g2 = np.array([f["g2"] for _, f in ok])
        h0 = [f for h, f in ok if h == 0]
        best = global_fit(hs, g2)  # 2026-11-25: global search (was: least_squares from 4 starts)
        g0, c = best.x
        hstar = -g0 / c
        rms = math.sqrt(2 * best.cost / len(hs))
        fits[(K, chi)] = (g0, c, hstar, rms)
        ne0 = h0[0]["ne"] if h0 else None
        near = min(ok, key=lambda t: abs(t[0] - hstar))
        print(f"K={K} chi={chi}: g_seed(12)={g0:+.3e} (h=0 run |g|={math.sqrt(h0[0]['g2']) if h0 else float('nan'):.3e}), "
              f"c={c:.3f}, h*={hstar:+.2e}, rms log residual {rms:.2f}; n_e(h=0)={ne0}, "
              f"closest h={near[0]:+.1e}: g2={near[1]['g2']:.2e}, n_e={near[1]['ne']}, max n_e={max(f['ne'] for _, f in ok)}")
        for h, f in pts:
            if f:
                print(f"    h={h:+.1e}: g2={f['g2']:.3e} (model {(g0 + c * h) ** 2:.3e}), n_e={f['ne']}, r/2^(2-K)={f['r'] / 2 ** (2 - K):.4f}")
                rows.append(dict(K=K, chi=chi, h=h, g2_n12=f["g2"], n_e=f["ne"], r_over_pred=f["r"] / 2 ** (2 - K),
                                 g_seed=g0, c=c, h_star=hstar))
            else:
                print(f"    h={h:+.1e}: no breakdown yet / no window")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(HERE, "b9.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    figure(runs, fits)


def figure(runs, fits):
    """(a) g_d^2(12) / g_d^2(12; h=0) and (b) n_e - n_e(h=0) versus h/h*, all (K, chi) on common axes."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts (journals reject Type 3)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    Ks = sorted({K for K, _ in fits}); cmap = plt.get_cmap("viridis")
    for (K, chi), pts in sorted(runs.items()):
        if (K, chi) not in fits:
            continue
        ok = sorted((h, f) for h, f in pts if f)
        h0 = [f for h, f in ok if h == 0]
        if not h0:
            continue
        g0, c, hstar, _ = fits[(K, chi)]
        col = cmap(Ks.index(K) / max(len(Ks) - 1, 1)); mk = "o" if chi == 32 else "s"
        u = np.array([h / hstar for h, _ in ok])
        ax[0].plot(u, [f["g2"] / h0[0]["g2"] for _, f in ok], mk, ms=3, color=col, mfc="none" if chi == 40 else col,
                   label=rf"$K={K:g}$" if chi == 32 else None)
        ax[1].plot(u, [f["ne"] - h0[0]["ne"] for _, f in ok], mk + "-", ms=3, lw=0.6, color=col, mfc="none" if chi == 40 else col)
    uu = np.linspace(-1.2, 2.2, 400)
    ax[0].plot(uu, (1 - uu) ** 2, "k-", lw=0.8, label=r"$(1-\mu/\mu^*)^2$")
    ax[0].set_yscale("log"); ax[0].set_ylim(1e-4, 30); ax[0].set_xlim(-1.2, 2.2)
    ax[0].set_xlabel(r"$\mu/\mu^*$"); ax[0].set_ylabel(r"$g_{\rm d}^2(12)\,/\,g_{\rm d}^2(12)|_{\mu=0}$")
    ax[0].legend(fontsize=6, frameon=False, ncol=2); ax[0].set_title(r"(a) seed (filled: $\chi=32$, open: $\chi=40$)", fontsize=8.5)
    ax[1].axvline(1, color="k", ls=":", lw=0.8); ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlim(-1.2, 2.2); ax[1].set_xlabel(r"$\mu/\mu^*$"); ax[1].set_ylabel(r"$n_e-n_e|_{\mu=0}$")
    ax[1].set_title(r"(b) delay of the breakdown", fontsize=8.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_b9.{ext}"), dpi=200)


if __name__ == "__main__":
    main(sys.argv[1])
