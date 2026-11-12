"""ED check of the first-order twisted splitting, paper 1 Eq. (23) (2026-11-12, Yamazaki).

Dimerized XXZ chain H = sum_j (1 + d (-1)^j) [Sx Sx + Sy Sy + Delta Sz Sz], Delta = -cos(pi/(2K)), even L.
pi twist: the XY part of the boundary bond changes sign.  Tested, without any CFT normalization of d:
  (i)  at d = 0 the two lowest twisted Sz=0 levels sit at x = K/4 (charge shift e -> e + 1/2);
  (ii) they split linearly in d, delta_pi = a1 d, and the lowest charged level shifts as b2 d^2;
       Eq. (23) with Eq. (B5) predicts a1^2 / b2 = (2 sqrt2 pi)^2 / alpha_{0,1}(K) = 8 pi^2 / alpha_{0,1}.
Scale: L/(2 pi v) from x_{0,1}(d=0) = 1/(4K).  C = spin flip.
"""
import itertools, sys
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
from math import comb, cos, pi

def basis(L, nup):
    st = [sum(1 << i for i in c) for c in itertools.combinations(range(L), nup)]
    return np.array(st, dtype=np.int64), {s: i for i, s in enumerate(st)}

def ham(L, nup, d, Delta, twist):
    st, idx = basis(L, nup); N = len(st)
    rows, cols, vals = [], [], []
    diag = np.zeros(N)
    for j in range(L):
        k = (j + 1) % L; J = 1 + d * (-1) ** j
        sxy = -1.0 if (twist and k == 0) else 1.0
        bj = (st >> j) & 1; bk = (st >> k) & 1
        diag += J * Delta * np.where(bj == bk, 0.25, -0.25)
        flip = bj != bk
        new = st[flip] ^ ((1 << j) | (1 << k))
        rows += [idx[s] for s in new]; cols += list(np.nonzero(flip)[0]); vals += [0.5 * J * sxy] * int(flip.sum())
    H = sp.csr_matrix((vals, (rows, cols)), shape=(N, N)) + sp.diags(diag)
    return H, st, idx

def lowest(H, k):
    return np.sort(sla.eigsh(H, k=k, which='SA', tol=1e-12)[0])

def flip_parity_levels(H, st, idx, L, k=4):
    w, v = sla.eigsh(H, k=k, which='SA', tol=1e-12)
    full = (1 << L) - 1; perm = np.array([idx[s ^ full] for s in st])
    order = np.argsort(w); out = []
    for i in order:
        x = v[:, i]; out.append((w[i], float(np.real(np.vdot(x, x[perm])))))
    return out

def run(K, L, ds=(2e-3, 4e-3)):
    Delta = -cos(pi / (2 * K)); n0 = L // 2
    def data(d):
        H0, *_ = ham(L, n0, d, Delta, False); E0 = lowest(H0, 1)[0]
        H1, *_ = ham(L, n0 + 1, d, Delta, False); E1 = lowest(H1, 1)[0]
        Ht, st, idx = ham(L, n0, d, Delta, True); lv = flip_parity_levels(Ht, st, idx, L)
        return E0, E1, lv
    E0, E1, lv0 = data(0.0)
    scale = 1 / (4 * K * (E1 - E0))           # L/(2 pi v)
    xt = [(scale * (e - E0), p) for e, p in lv0[:2]]
    a1s, b2s = [], []
    for d in ds:
        Ep0, Ep1, lvp = data(d); Em0, Em1, lvm = data(-d)
        def split(lv, e0):
            two = lv[:2]; ev = [e for e, p in two if p > 0]; od = [e for e, p in two if p < 0]
            return scale * (od[0] - ev[0]) if ev and od else np.nan
        sp_ = split(lvp, Ep0); sm = split(lvm, Em0)
        a1s.append((sp_ - sm) / (2 * d))
        gap = lambda a, b: scale * (b - a)
        b2s.append((gap(Ep0, Ep1) + gap(Em0, Em1) - 2 * gap(E0, E1)) / (2 * d * d))
    a1 = (4 * a1s[0] - a1s[1]) / 3; b2 = (4 * b2s[0] - b2s[1]) / 3   # Richardson
    return xt, a1, b2

if __name__ == '__main__':
    alpha = {1.0: 27.364, 1.5: 17.089, 1.25: 20.320}
    for K in map(float, sys.argv[1].split(',')):
        for L in map(int, sys.argv[2].split(',')):
            xt, a1, b2 = run(K, L)
            print(f"K={K} L={L}  twisted x(d=0)={xt[0][0]:.5f}({xt[0][1]:+.0f}),{xt[1][0]:.5f}({xt[1][1]:+.0f}) [K/4={K/4:.5f}]"
                  f"  a1^2/b2={a1*a1/b2:.4f}  pred 8pi^2/alpha={8*pi*pi/alpha[K]:.4f}  sign(a1)={np.sign(a1):+.0f}", flush=True)
