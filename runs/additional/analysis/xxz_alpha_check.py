"""Independent lattice check of the parameter-free amplitude ratio alpha_J / (4 K alpha_{0,1}) (2026-11-06).

Exact diagonalization of the periodic XXZ chain with bond alternation,
    H = sum_j [1 + (-1)^j delta] (S^x_j S^x_{j+1} + S^y_j S^y_{j+1} + Delta S^z_j S^z_{j+1}),
with Delta = -cos(pi / (2K)) (K = 1/2 at the SU(2) point, K = 1 at the XX point), L = 12, 16, 20, 24.
Levels x = (E - E_0) L / (2 pi v), v = pi sin(gamma) / (2 gamma), cos(gamma) = Delta:
    X   = 1 - 4K x_{0,1}   (lowest S^z = 1 state)
    X_J = 1 - x_J          (S^z = 0 pair with translation-by-two eigenvalue exp(+-4 pi i / L): the currents).
The dimerization coupling g_d is proportional to delta L^{2-K}; the second-order shifts are
    dX_i(L) / delta^2 = a_i + b_i L^{2(2-K)},
where a_i (L independent) collects analytic effects such as the change of the velocity.  CFT predicts
    b_J / b_X = alpha_J / (4 K alpha_{0,1})     (Eq. (ampratio) of the paper; 1/(4 ln 2) at K = 1).
The delta^2 coefficient is obtained by Richardson extrapolation from delta and 2 delta.
Usage:  python3 xxz_alpha_check.py [K ...]
"""
import sys, math, itertools
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

EPS = 1e-3
ALPHA = {0.75: (40.640, 47.840), 1.0: (27.364, 39.478), 1.25: (20.320, 32.578), 1.5: (17.089, 27.059)}  # (alpha01, alphaJ), Table VII


def basis(L, nup):
    states = np.array(sorted(sum(1 << i for i in c) for c in itertools.combinations(range(L), nup)), dtype=np.int64)
    return states


def hamiltonian(L, states, Delta, delta):
    rows, cols, vals = [], [], []
    diag = np.zeros(len(states))
    for j in range(L):
        k = (j + 1) % L
        w = 1 + (-1) ** j * delta
        bj = (states >> j) & 1; bk = (states >> k) & 1
        diag += w * Delta * np.where(bj == bk, 0.25, -0.25)
        flip = np.nonzero(bj != bk)[0]
        t = states[flip] ^ ((1 << j) | (1 << k))
        rows.append(np.searchsorted(states, t)); cols.append(flip); vals.append(np.full(len(flip), 0.5 * w))
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    H = sp.csr_matrix((vals, (rows, cols)), shape=(len(states), len(states)))
    return H + sp.diags(diag)


def translate2(L, states):
    mask = (1 << L) - 1
    t = ((states << 2) | (states >> (L - 2))) & mask
    order = np.searchsorted(states, t)
    return order   # T2 maps basis state i to basis state order[i]


def lowest(H, k):
    vals, vecs = eigsh(H, k=k, which="SA", tol=1e-13)
    o = np.argsort(vals)
    return vals[o], vecs[:, o]


def levels(L, Delta, delta):
    s0 = basis(L, L // 2); s1 = basis(L, L // 2 + 1)
    # T2 commutes with H.  Adding EPS (T2 + T2^dagger) lifts accidental degeneracies between different T2 sectors
    # (exact at the XX point between the currents and the e = +-1 states), so that eigenvectors are T2 eigenstates;
    # the added energy EPS <T2 + T2^dagger> is subtracted exactly afterwards.
    T2 = translate2(L, s0)
    P = sp.csr_matrix((np.ones(len(s0)), (T2, np.arange(len(s0)))), shape=(len(s0), len(s0)))
    e0, v0 = lowest(hamiltonian(L, s0, Delta, delta) + EPS * (P + P.T), 10)
    t2 = np.array([np.dot(v0[T2, a], v0[:, a]) for a in range(v0.shape[1])])  # Re <psi|T2|psi> (real states)
    e0 = e0 - 2 * EPS * t2
    e1, _ = lowest(hamiltonian(L, s1, Delta, delta), 2)
    cJ = math.cos(4 * math.pi / L)
    J = [e for e, t in zip(e0[1:], t2[1:]) if abs(t - cJ) < 0.02]
    if len(J) < 2:
        raise RuntimeError(f"current pair not found at L={L}: T2 = {t2}")
    return e0[0], e1[0], np.mean(J[:2])


def main(Ks):
    for K in Ks:
        Delta = -math.cos(math.pi / (2 * K)); gam = math.acos(Delta); v = math.pi * math.sin(gam) / (2 * gam)
        a01, aJ = ALPHA[K]; pred = aJ / (4 * K * a01)
        Ls = [12, 16, 20, 24]; d = 2e-3
        dX, dXJ = [], []
        for L in Ls:
            sc = L / (2 * math.pi * v)
            out = {}
            for dd in (0.0, d, 2 * d):
                E0, E1, EJ = levels(L, Delta, dd)
                out[dd] = (1 - 4 * K * (E1 - E0) * sc, 1 - (EJ - E0) * sc)
            for i, lst in ((0, dX), (1, dXJ)):
                f1 = out[d][i] - out[0.0][i]; f2 = out[2 * d][i] - out[0.0][i]
                lst.append((16 * f1 - f2) / (12 * d * d))
            print(f"K={K} L={L}: X(0)={out[0.0][0]:+.4f} XJ(0)={out[0.0][1]:+.4f}  "
                  f"dX/d^2={dX[-1]:+.4f} dXJ/d^2={dXJ[-1]:+.4f}", flush=True)
        p = 2 * (2 - K); Lp = np.array(Ls, float) ** p
        M = np.vstack([np.ones_like(Lp), Lp]).T
        (aX, bX), *_ = np.linalg.lstsq(M, np.array(dX), rcond=None)
        (aJf, bJ), *_ = np.linalg.lstsq(M, np.array(dXJ), rcond=None)
        # same fit without the smallest size, as a finite-size check
        (_, bX3), *_ = np.linalg.lstsq(M[1:], np.array(dX[1:]), rcond=None)
        (_, bJ3), *_ = np.linalg.lstsq(M[1:], np.array(dXJ[1:]), rcond=None)
        print(f"== K={K}: b_J/b_X = {bJ / bX:.4f} (L=12-24), {bJ3 / bX3:.4f} (L=16-24); prediction {pred:.4f}; "
              f"a_X={aX:+.3f} a_J={aJf:+.3f}", flush=True)


if __name__ == "__main__":
    main([float(k) for k in sys.argv[1:]] or [1.0, 1.5, 1.25, 0.75])
