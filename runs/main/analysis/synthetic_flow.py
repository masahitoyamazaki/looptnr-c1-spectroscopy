"""Estimate how the running of K (flow equations) biases the O(2) fits.

No new RG or CFT measurement: the only inputs are the fitted growth
amplitudes in o2_fits.csv and the coefficients in
sixvertex_g2_alpha.csv.

For each run, the one-step recursion (Otsuki, perturbation notes, "flow")
    g_{n+1} = 2^{(2-K_n)/2} g_n
    K_{n+1} = K_n - pi^2 K_n^2 (2^{2-K_n}-1)/(2-K_n) g_n^2
is started at n=12 with g chosen such that the synthetic charged-level
deviation at n_e equals the measured one.  The synthetic levels are
    X   = -4K [alpha_reg g^2 - dK/(4K^2)],   alpha_reg = alpha_01 - pi^2/(4(2-K))
    X_J = -alpha_J g^2                        (no K dependence of x_J)
i.e. the 1/(2-K) pole of alpha_01, which represents the induced shift of K at
fixed K, is replaced by the explicitly running dK.  Both are then fitted in
the same way as the data (c + d t + A r^{n-n_e}, n=12..n_e) and r/2^{2-K}
and (A_J/A)/[alpha_J/(4K alpha_01)] are printed next to the measured values.

Paper I, App. D ("Bias of the growth rate", "Amplitude ratio near K=2"), inputs o2_fits.csv and sixvertex_g2_alpha.csv.
Usage: python3 synthetic_flow.py   (after analyze.py)
"""
import csv
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ALPHA = HERE / "sixvertex_g2_alpha.csv"
FITS = HERE / "o2_fits.csv"
N_START = 12


def fit_rate(n, x, n_end):
    t = (n - (N_START + n_end) / 2) / 10
    best = None
    for log_r in np.linspace(math.log(1.02), math.log(4), 8000):
        m = np.vstack([np.ones_like(t), t, np.exp(log_r * (n - n_end))]).T
        coef, *_ = np.linalg.lstsq(m, x, rcond=None)
        rss = ((m @ coef - x) ** 2).sum()
        if best is None or rss < best[0]:
            best = (rss, log_r, coef, m)
    return best[1:]


def flow(g12, k0, n_end):
    g, k, out = g12, k0, []
    for n in range(N_START, n_end + 1):
        out.append((n, g * g, k))
        dk = -math.pi**2 * k**2 * (2 ** (2 - k) - 1) / (2 - k) * g * g
        g, k = 2 ** ((2 - k) / 2) * g, k + dk
    return out


def main():
    alpha = {round(float(r["K"]), 4): r for r in csv.DictReader(open(ALPHA))}
    print("chi     K   r/2^(2-K): sim  meas   ratio/pred: sim  meas")
    for row in csv.DictReader(open(FITS)):
        if not row["n_e"] or not row["A"] or float(row["K"]) not in (0.75, 1.0, 1.25, 1.5, 1.75):
            continue
        k0, n_end = float(row["K"]), int(row["n_e"])
        a = alpha[round(k0, 4)]
        a01, a_j = float(a["m1"]), float(a["J"])
        a_reg = a01 - math.pi**2 / (4 * (2 - k0))
        x_end = float(row["A"])  # measured growth amplitude at n_end

        def x_of(state):
            _, g2, k = state
            return -4 * k0 * (a_reg * g2 - (k - k0) / (4 * k0**2))

        lo, hi = 1e-10, 1e-1
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            if x_of(flow(mid, k0, n_end)[-1]) > x_end:
                lo = mid
            else:
                hi = mid
        states = flow(mid, k0, n_end)
        n = np.array([s[0] for s in states], float)
        x = np.array([x_of(s) for s in states])
        x_j = -a_j * np.array([s[1] for s in states])
        log_r, coef, m = fit_rate(n, x, n_end)
        coef_j, *_ = np.linalg.lstsq(m, x_j, rcond=None)
        ratio = (coef_j[2] / coef[2]) / (a_j / (4 * k0 * a01))
        pred = 2 ** (2 - k0)
        print(f"{row['chi']:>3} {k0:5.2f}   {math.exp(log_r)/pred:.4f} {float(row['r'])/pred:.4f}"
              f"   {ratio:.4f} {float(row['ratio_over_pred']):.4f}")


if __name__ == "__main__":
    main()
