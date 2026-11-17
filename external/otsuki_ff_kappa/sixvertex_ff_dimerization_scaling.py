# Exact finite-size scaling function of the lowest charged level of the dimerized XX chain (K = 1, free fermions),
# used to predict the fourth-order (g_d^4) correction to the O(2) growth fit of paper 1a at K = 1.
#
# H = J sum_j (1 + (-1)^j delta) (S^x_j S^x_{j+1} + S^y_j S^y_{j+1}), periodic, L sites (L = 0 mod 4), J = 1.
# Jordan-Wigner: hopping t_j = (1 + (-1)^j delta) / 2; the fermion boundary condition is antiperiodic for even N
# and periodic for odd N. Two-site cells, cell momentum q = (2 pi m + theta) / (L/2), bands +-|t1 + t2 e^{iq}|.
# Ground state of the N-fermion sector: fill the lowest N single-particle levels.
#
# Levels: x_n(L, delta) = L (E_n - E_0) / (2 pi v), v = J = 1 (velocity at delta = 0).
#   m1: S^z = 1 sector (N = L/2 + 1), CFT value 1/(4K) = 1/4;  m2: S^z = 2 (N = L/2 + 2), CFT value 1/K = 1.
# Scaling variable s = delta * L (g_d is proportional to s at K = 1). X = 1 - 4K x_m1 = 1 - 4 x_m1.
#
# Checks: x -> 1/4 and 1 at delta = 0; the ratio of the s^2 coefficients of m2 and m1 equals
# alpha_{0,2} / alpha_{0,1} = 4 pi^2 / (4 pi^2 ln 2) = 1 / ln 2 (paper App. B, free-fermion values);
# the scaling function does not depend on L.
#
# Output: Otsuki/sixvertex/data/sixvertex_ff_dimerization_scaling.csv (s, X, x_m2 shift for several L)
#         Otsuki/sixvertex/logs/sixvertex_ff_dimerization_scaling.log (coefficients a, b of X = a s^2 + b s^4 + ...,
#         and kappa = b / a^2, i.e. X = X_g + kappa X_g^2 + ... with X_g = a s^2)
#
# Usage (from the repository root):
#   uv run --no-project --with numpy python external/otsuki_ff_kappa/sixvertex_ff_dimerization_scaling.py

import csv
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE
OUT_DATA = HERE
OUT_LOG = HERE
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def single_particle(L, delta, theta):
    Lc = L // 2
    q = (2 * np.pi * np.arange(Lc) + theta) / Lc
    t1, t2 = (1 + delta) / 2, (1 - delta) / 2
    e = np.abs(t1 + t2 * np.exp(1j * q))
    return np.sort(np.concatenate([-e, e]))


def energy(L, delta, N):
    theta = math.pi if N % 2 == 0 else 0.0
    eps = single_particle(L, delta, theta)
    return float(np.sum(eps[:N]))


def levels(L, delta):
    N0 = L // 2
    E0 = energy(L, delta, N0)
    x1 = L * (energy(L, delta, N0 + 1) - E0) / (2 * math.pi)
    x2 = L * (energy(L, delta, N0 + 2) - E0) / (2 * math.pi)
    return x1, x2


def poly_coeffs(s, y, deg):
    """Fit y = sum_{k=1..deg} c_k s^{2k} (no constant), return c_1..c_deg."""
    A = np.column_stack([s ** (2 * k) for k in range(1, deg + 1)])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c


def main():
    OUT_DATA.mkdir(exist_ok=True)
    OUT_LOG.mkdir(exist_ok=True)
    rows = []
    log("# dimerized XX chain (K = 1): exact finite-size levels as functions of s = delta L")
    for L in (1024, 4096, 16384):
        x1, x2 = levels(L, 0.0)
        log(f"L = {L}: delta = 0: x_m1 = {x1:.8f} (1/4), x_m2 = {x2:.8f} (1)")
    s_grid = np.concatenate([np.linspace(0.005, 0.2, 40), np.linspace(0.21, 3.0, 120)])
    res = {}
    for L in (1024, 4096, 16384):
        x10, x20 = levels(L, 0.0)
        X = []
        d2 = []
        for s in s_grid:
            x1, x2 = levels(L, s / L)
            X.append(1 - 4 * x1)
            d2.append(x2 - x20)
            rows.append(dict(L=L, s=s, X=1 - 4 * x1, X0=1 - 4 * x10, dx_m2=x2 - x20))
        res[L] = (np.array(X) - (1 - 4 * x10), np.array(d2))
    # coefficients from small s (s <= 0.6), with increasing polynomial order as a stability check
    for L in (1024, 4096, 16384):
        dX, d2 = res[L]
        sel = s_grid <= 0.6
        for deg in (3, 4, 5):
            c = poly_coeffs(s_grid[sel], dX[sel], deg)
            c2 = poly_coeffs(s_grid[sel], d2[sel], deg)
            a, b = c[0], c[1]
            ratio = c2[0] / (-c[0] / 4)          # (s^2 coeff of dx_m2) / (s^2 coeff of dx_m1), dx_m1 = -dX/4
            log(f"L = {L}, deg = {deg}: X = a s^2 + b s^4 + ...: a = {a:.8f}, b = {b:.8f}, kappa = b/a^2 = {b / a ** 2:.6f}; "
                f"m2/m1 s^2 ratio = {ratio:.6f} (1/ln2 = {1 / math.log(2):.6f})")
    with open(OUT_DATA / "sixvertex_ff_dimerization_scaling.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT_LOG / "sixvertex_ff_dimerization_scaling.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
