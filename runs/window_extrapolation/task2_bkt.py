"""Task 2: dimerization growth rate at the BKT point K = 1/2 versus the log-corrected expectation.

Usage: python3 task2_bkt.py   -> task2_ghat_steps.csv, task2_bkt.csv (and printed tables)

Measured rate r: the paper's protocol (analyze.analyze_o2 / Sec. IV B): X = 1 - 4K x_{0,1} = 1 - 2 x_{0,1},
window n = 12 .. n_e (n_e = step before |X| first exceeds 0.1), columns 1, t, r^(n-n_e), umklapp (2^(2-4K))^(n-12)
(which is a constant at K = 1/2, i.e. degenerate with the constant column; lstsq returns the min-norm solution).
Smaller cuts are not usable at K = 1/2: X(12) = ghat/2 ~ 0.074 already, so |X| <= 0.05 leaves no window.

ghat(n): from the SU(2) triplet, x_t = 1/2 - ghat/4 (manuscript Eq. eq:bkt), with x_t the average of the lowest
|m| = 1 level [(1, 2) sector] and the lowest C-odd neutral level [(0, 1) sector], exactly as in
additional/analysis/b8.py.  Late in the window the triplet is also shifted by the growing dimerization
(g_d^2), so the direct value ghat_t(n) = 4(1/2 - x_t(n)) is biased there.  We therefore use three estimates:
  flow : two-loop flow dghat/dl = -ghat^2 - ghat^3/2, l = n ln sqrt2 + const, with the single constant fitted to
         ghat_t(n) over the clean range n = 8..20 and integrated into the window (primary);
  lin  : empirical straight-line fit of 1/ghat_t(n) = a + b n over n = 8..20, extrapolated (accounts for the
         measured slope 1.49-1.55 of 4/ghat per step, which exceeds the two-loop value);
  direct: ghat_t(n) measured at each step (contaminated by g_d^2 near n_e; shown for reference only).
Singlet cross-check: ghat_s = (4/3)(x_s - 1/2), x_s = second level of the (0, 0) sector.
Log-corrected expectation: mean over n = 12..n_e of 2^(3/2 - 3 ghat(n)/4)  (plain average, as specified);
also its min/max over the window and the value averaged over the last 5 window steps, where the growth term
dominates X.  Supplementary fits (same window): background {linbg: 1, t (paper); ghatbg: X - ghat_flow(n)/2 fitted with 1, t
(X = 1 - 2 x_t = ghat/2 by eq:bkt); freeghatbg: 1, t, ghat_flow(n) with free coefficient} x growth column
{plain: r^(n-n_e); logcol: prod_m [rho 2^(-3 ghat_flow(m)/4)], value rho/2^(3/2) = measured / log-corrected}.
"""
import csv, math, os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from common import HERE, series, fit_cut

CHIS = (24, 32, 40, 48, 64)
L2 = math.log(math.sqrt(2))
CLEAN = {24: (8, 14)}            # chi = 24: dimerization visible from n ~ 16
CLEAN_DEFAULT = (8, 20)


def levels(lev, n):
    rows = lev[n]
    ch = sorted(float(r["x"]) for r in rows if "(1, 2)" in r["sector"])
    od = sorted(float(r["x"]) for r in rows if "(0, 1)" in r["sector"])
    ev = sorted(float(r["x"]) for r in rows if "(0, 0)" in r["sector"])
    return 0.5 * (ch[0] + od[0]), ev[1]


def flow(g0, n0, ns):
    """two-loop flow from g(n0) = g0 to the steps ns (forward and backward)."""
    f = lambda l, g: -g ** 2 - 0.5 * g ** 3
    out = {}
    for sgn in (1, -1):
        tgt = sorted([n for n in ns if (n - n0) * sgn >= 0], key=lambda n: sgn * n)
        if not tgt:
            continue
        s = solve_ivp(f, (0, sgn * (max(abs(n - n0) for n in tgt) + 1e-9) * L2), [g0],
                      t_eval=[(n - n0) * L2 for n in tgt], rtol=1e-11, atol=1e-13)
        out.update(dict(zip(tgt, s.y[0])))
    return np.array([out[n] for n in ns])


def main():
    steps_rows, rows = [], []
    for chi in CHIS:
        X, XJ, lev, _ = series(0.5, chi)
        f = fit_cut(0.5, X, XJ, 0.1)
        ne = f["ne"]
        ns_all = [n for n in sorted(lev) if 4 <= n <= ne + 2]
        gt = {n: 4 * (0.5 - levels(lev, n)[0]) for n in ns_all}
        gs = {n: 4 / 3 * (levels(lev, n)[1] - 0.5) for n in ns_all}
        c0_, c1_ = CLEAN.get(chi, CLEAN_DEFAULT)
        cl = [n for n in ns_all if c0_ <= n <= c1_]
        n0 = 14
        res = minimize_scalar(lambda g0: ((flow(g0, n0, cl) - np.array([gt[n] for n in cl])) ** 2).sum(),
                              bounds=(0.05, 0.3), method="bounded", options=dict(xatol=1e-10))
        win = list(range(12, ne + 1))
        g_flow = flow(res.x, n0, win)
        rms_flow = math.sqrt(res.fun / len(cl))
        p = np.polyfit(cl, [1 / gt[n] for n in cl], 1)
        g_lin = 1 / np.polyval(p, win)
        g_dir = np.array([gt[n] for n in win])
        for i, n in enumerate(win):
            steps_rows.append(dict(chi=chi, n=n, X=X[n], ghat_triplet_direct=gt[n], ghat_singlet_direct=gs[n],
                                   ghat_flow=g_flow[i], ghat_lin=g_lin[i]))
        row = dict(chi=chi, window=f"12-{ne}", npts=f["npts"], r=f["r"], r_over_2_32=f["r"] / 2 ** 1.5,
                   flow_g14=res.x, flow_rms_n8_20=rms_flow, lin_slope_4_over_g=4 * p[0])
        for tag, g in (("flow", g_flow), ("lin", g_lin), ("direct", g_dir)):
            lam = 2 ** (1.5 - 0.75 * g)
            row.update({f"ghat_{tag}_n12": g[0], f"ghat_{tag}_ne": g[-1], f"ghat_{tag}_mean": g.mean(),
                        f"expect_{tag}_mean": lam.mean(), f"expect_{tag}_min": lam.min(), f"expect_{tag}_max": lam.max(),
                        f"expect_{tag}_last5": lam[-5:].mean(),
                        f"r_over_expect_{tag}": f["r"] / lam.mean(),
                        f"r_over_expect_{tag}_last5": f["r"] / lam[-5:].mean()})
        # supplementary fits over the same window: background model x growth form
        n = np.array(win, float)
        t = (n - (12 + ne) / 2) / 10
        x = np.array([X[m] for m in win])
        cum = np.cumsum(-0.75 * g_flow * math.log(2))
        cum -= cum[-1]                                       # sum_{m<=n} ln 2^{-3g/4}, zero at n_e
        bgs = {"linbg": (x, [np.ones_like(t), t]),                         # paper: 1, t (umklapp col = const)
               "ghatbg": (x - g_flow / 2, [np.ones_like(t), t]),           # X - ghat_flow/2 (X = ghat/2 by eq:bkt)
               "freeghatbg": (x, [np.ones_like(t), t, g_flow])}           # free coefficient of ghat_flow
        for bname, (y, cols) in bgs.items():
            for gname, lc in (("plain", 0.0), ("logcol", 1.0)):
                best = None
                for lr in np.linspace(math.log(1.02), math.log(4), 4000):
                    M = np.vstack(cols + [np.exp(lr * (n - ne) + lc * cum)]).T
                    c, *_ = np.linalg.lstsq(M, y, rcond=None)
                    rss = ((M @ c - y) ** 2).sum()
                    if best is None or rss < best[0]:
                        best = (rss, lr, c)
                key = f"{bname}_{gname}"
                row[f"{key}_rate_over_2_32"] = math.exp(best[1]) / 2 ** 1.5
                row[f"{key}_rms"] = math.sqrt(best[0] / len(n))
                if bname == "freeghatbg":
                    row[f"{key}_ghat_coef"] = best[2][2]
        row["ghatbg_plain_r_over_expect_flow"] = row["ghatbg_plain_rate_over_2_32"] * 2 ** 1.5 / row["expect_flow_mean"]
        rows.append(row)
    write(os.path.join(HERE, "task2_ghat_steps.csv"), steps_rows)
    write(os.path.join(HERE, "task2_bkt.csv"), rows)
    print("chi window  r      r/2^1.5 | ghat_flow n12..ne (mean)   <2^(3/2-3g/4)>  r/<.> | lin: ghat range, r/<.> | "
          "direct: r/<.> | last5(flow) r/<.> | linbg-logcol rho/2^1.5")
    for r in rows:
        print(f"{r['chi']:3d} {r['window']:6s} {r['r']:.4f} {r['r_over_2_32']:.4f} | "
              f"{r['ghat_flow_n12']:.4f}..{r['ghat_flow_ne']:.4f} ({r['ghat_flow_mean']:.4f}) "
              f"{r['expect_flow_mean']:.4f} {r['r_over_expect_flow']:.4f} | "
              f"{r['ghat_lin_n12']:.4f}..{r['ghat_lin_ne']:.4f} {r['r_over_expect_lin']:.4f} | "
              f"{r['ghat_direct_n12']:.4f}..{r['ghat_direct_ne']:.4f} {r['r_over_expect_direct']:.4f} | "
              f"{r['r_over_expect_flow_last5']:.4f} | {r['linbg_logcol_rate_over_2_32']:.4f}   "
              f"(flow rms {r['flow_rms_n8_20']:.1e}, 4/g slope {r['lin_slope_4_over_g']:.3f})")
    print("\nBackground-model check (rate/2^1.5; 'logcol' = per-step factor rho*2^(-3 ghat_flow/4), value is rho/2^1.5):")
    for r in rows:
        print(f"{r['chi']:3d} " + " | ".join(f"{b}: plain {r[b+'_plain_rate_over_2_32']:.4f} logcol {r[b+'_logcol_rate_over_2_32']:.4f} "
                                          f"rms {r[b+'_plain_rms']:.1e}" for b in ("linbg", "ghatbg", "freeghatbg"))
              + f" | free ghat coef {r['freeghatbg_plain_ghat_coef']:.3f} | ghatbg plain r/<expect> {r['ghatbg_plain_r_over_expect_flow']:.4f}")


def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{float(v):.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
