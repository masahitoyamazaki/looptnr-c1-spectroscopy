"""Independent check of the prediction chain g_d -> mu_SG -> M at the free-fermion point K = 1.

Dimerized XX chain  H = sum_j (1 + delta (-1)^j) (S^x_j S^x_{j+1} + S^y_j S^y_{j+1}),  periodic spins,
mapped to free fermions (hopping (1 +- delta)/2; boundary sign (-1)^(N+1)).  Velocity v = 1 at delta = 0.

1. Second order: x_{0,1}(L, delta) - x_{0,1}(L, 0) = alpha_{0,1} g~^2 (L / 2 pi)^2 with alpha_{0,1}(K=1) = 4 pi^2 ln 2
   (paper 1), which fixes g~ = c delta.
2. Sine-Gordon (Zamolodchikov, Eq. (zam) of the paper) with |g~| = sqrt2 mu_SG predicts the soliton mass
   M = sqrt2 pi |g~| at K = 1 (p = 1, kappa = 1/(2 pi)).
3. Exact: the single-particle gap of the dimerized chain is delta (v = 1), so M_exact = delta.
The ratio M_pred / M_exact = sqrt2 pi c must be 1.

Also checks numerically that Zamolodchikov's kappa(p) and the Takayoshi-Sato form used by Yamamoto give the same M.
Usage: ~/miniforge3/bin/python scripts/check_free_fermion_K1.py
"""
import math
import numpy as np


def levels(L, delta, periodic):
    h = np.zeros((L, L))
    for j in range(L - 1):
        t = 0.5 * (1 + delta * (-1) ** (j + 1))      # bond (j+1, j+2) in 1-based sites, sign (-1)^(j+1)
        h[j, j + 1] = h[j + 1, j] = t
    tb = 0.5 * (1 + delta * (-1) ** L)              # bond (L, 1)
    s = 1.0 if periodic else -1.0
    h[L - 1, 0] = h[0, L - 1] = s * tb
    return np.linalg.eigvalsh(h)


def x01(L, delta):
    """Lowest |m| = 1 level: N = L/2 + 1 fermions (odd -> periodic) against N = L/2 (even -> antiperiodic)."""
    assert L % 4 == 0
    e_ap = levels(L, delta, periodic=False)
    e_p = levels(L, delta, periodic=True)
    E0 = e_ap[: L // 2].sum()
    E1 = e_p[: L // 2 + 1].sum()
    return L / (2 * math.pi) * (E1 - E0)


def kappa(p):
    return (math.gamma(p / (p + 1)) / math.gamma(1 / (p + 1)) / math.pi
            * (math.sqrt(math.pi) * math.gamma((p + 1) / 2) / (2 * math.gamma(p / 2))) ** (2 / (p + 1)))


def mass_zam(K, gtilde):
    p = K / (2 - K)
    return (abs(gtilde) / (math.sqrt(2) * kappa(p))) ** (1 / (2 - K))


def mass_ts(K, gtilde):
    """Form used by Yamamoto: M = C_K [B_K |g~|]^(1/(2-K))."""
    p = K / (2 - K)
    C = 2 * math.gamma(p / 2) / (math.sqrt(math.pi) * math.gamma((1 + p) / 2))
    B = (math.pi / math.sqrt(2)) * math.gamma(1 - K / 2) / math.gamma(K / 2)
    return C * (B * abs(gtilde)) ** (1 / (2 - K))


def main():
    print("Zamolodchikov vs Takayoshi-Sato form (relative difference of M):")
    for K in (0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 1.75):
        print(f"  K = {K:4}: {mass_zam(K, 1e-4) / mass_ts(K, 1e-4) - 1:+.2e}")
    print(f"kappa(1) = {kappa(1):.10f}, 1/(2 pi) = {1 / (2 * math.pi):.10f}")

    a01 = 4 * math.pi ** 2 * math.log(2)
    delta = 1e-5
    print("\nL      x01(0)      dx01/delta^2 / (L/2pi)^2 / alpha01 = c^2     sqrt2*pi*c")
    cs = []
    for L in (200, 400, 800, 1600, 3200):
        x0 = x01(L, 0.0)
        dx = x01(L, delta) - x0
        c2 = dx / delta ** 2 / (L / (2 * math.pi)) ** 2 / a01
        c = math.sqrt(c2)
        cs.append((L, c))
        print(f"{L:5d}  {x0:.8f}  {c2:.8e}  {math.sqrt(2) * math.pi * c:.6f}")
    # linear extrapolation in 1/L^2 of the last two points
    (L1, c1), (L2, c2_) = cs[-2], cs[-1]
    cinf = (c2_ * L2 ** 2 - c1 * L1 ** 2) / (L2 ** 2 - L1 ** 2)
    print(f"extrapolated sqrt2*pi*c = {math.sqrt(2) * math.pi * cinf:.6f}  (must be 1: M_pred / M_exact)")

    # exact infinite-chain gap: min over k of sqrt(cos^2 k + delta^2 sin^2 k) = delta
    k = np.linspace(0, math.pi, 200001)
    d = 0.01
    print(f"exact gap check at delta = {d}: {np.sqrt(np.cos(k) ** 2 + d ** 2 * np.sin(k) ** 2).min():.6f}")


if __name__ == "__main__":
    main()
