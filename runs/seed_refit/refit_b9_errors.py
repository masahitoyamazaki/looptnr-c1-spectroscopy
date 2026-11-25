"""Uncertainties of rho and g_seed of the compensation parabola (Paper 1b, Table I).

Same input and fit as refit_b9.py (seed read with rate 2^(2-K), n>=12, |X|<=cut, cut=0.02 by default).
  * se:  standard errors from the Jacobian of the least-squares fit in ln g2, scaled by the rms residual
  * jk:  jackknife (leave one run out) standard error
  * cut: rho and g_seed with the window cut 0.05 instead of 0.02 (readout dependence)
2026-11-25 fix: the fit (also inside the jackknife) is now global; the old least_squares from a few starts
had local minima, e.g. the leave-mu=0-out sample at (0.75,40).  Old outputs: old_20261125_localmin/.
Output: refit_b9_errors.csv
"""
import csv, math
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

SRC = Path(__file__).resolve().parents[3] / "Otsuki/sixvertex/data/sixvertex_sg_xi_predict.csv"
ALL = list(csv.DictReader(open(SRC)))

def groups(cut):
    g = {}
    for r in ALL:
        if r["cut"] == cut and r["rate"] == "fixed" and r["source"] == "Yamazaki B9" and r["g2_12"]:
            g.setdefault((float(r["K"]), int(r["chi"])), []).append((float(r["mu"]), float(r["g2_12"])))
    return g

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


fit = global_fit  # 2026-11-25: every fit, including each jackknife sample and the cut-0.05 fit, is global

G, G5 = groups("0.02"), groups("0.05")
out = []
for key, pts in sorted(G.items()):
    mu = np.array([p[0] for p in pts]); g2 = np.array([p[1] for p in pts])
    b = fit(mu, g2); gs, ka = b.x; n = len(mu)
    s2 = np.sum(b.fun ** 2) / max(n - 2, 1)
    cov = s2 * np.linalg.inv(b.jac.T @ b.jac); se_g, se_k = np.sqrt(np.diag(cov))
    jk = np.array([fit(np.delete(mu, i), np.delete(g2, i)).x for i in range(n)])
    jse = np.sqrt((n - 1) / n * np.sum((jk - jk.mean(0)) ** 2, axis=0))
    k5 = g5 = float("nan")
    if key in G5 and len(G5[key]) >= 3:
        p5 = G5[key]; b5 = fit(np.array([p[0] for p in p5]), np.array([p[1] for p in p5])); g5, k5 = b5.x
    out.append(dict(K=key[0], chi=key[1], n=n, rho=ka, rho_se=se_k, rho_jk=jse[1], rho_cut005=k5,
                    g_seed=gs, g_seed_se=se_g, g_seed_jk=jse[0], g_seed_cut005=g5, rms_ln_g2=math.sqrt(np.mean(b.fun ** 2))))
    o = out[-1]
    print(f"K={key[0]:.2f} chi={key[1]} n={n:2d} rho={ka:6.2f} se={se_k:.2f} jk={jse[1]:.2f} cut.05={k5:6.2f} | "
          f"g_seed={gs:+.2e} se={se_g:.1e} jk={jse[0]:.1e} cut.05={g5:+.2e} rms={o['rms_ln_g2']:.2f}")
w = csv.DictWriter(open(Path(__file__).with_name("refit_b9_errors.csv"), "w"), fieldnames=list(out[0]))
w.writeheader(); w.writerows(out)
