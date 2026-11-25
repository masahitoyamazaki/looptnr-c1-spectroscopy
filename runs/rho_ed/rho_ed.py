"""Independent lattice prediction of the response rho of Paper 1b (called kappa in early drafts) (Sec. IV B, Table III).

Diagonal-to-diagonal transfer matrix of the six-vertex model with weights (a,b,c)=(1,1,c), c=2 sin(pi/4K),
and the a/b alternation (a,b)=(1+mu,1-mu) on one sublattice and (1-mu,1+mu) on the other.
In the diagonal frame the vertices of one layer form one sublattice, so the two layers of the brickwork
carry the two signs of mu.  A row has N edges at spacing 1/sqrt2 (circumference L=N/sqrt2 in vertex
spacings a_0), and one double layer advances by sqrt2, so x = N/(4 pi) ln(lambda_0/lambda).

Readouts (normalization of Paper 1a, Eq. (coupling), L = circumference):
  * charged level  x_{0,1} -> 1/(4K)  (check of the gate convention)
  * pi-twisted neutral pair at K/4; first-order splitting delta_pi = x(C=-1)-x(C=+1) = -2 sqrt2 pi g_d
    (Paper 1b, Eq. (13)), i.e. |g_d(L)| = |delta_pi|/(2 sqrt2 pi).
Output: one CSV row per (K, N, mu, convention).
"""
import sys, csv, os, numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

def gate(a, b, c, conv):
    G = np.zeros((2, 2, 2, 2))  # G[out_i,out_j,in_i,in_j], bit 1 = up
    G[1, 1, 1, 1] = G[0, 0, 0, 0] = a
    stay, swap = (c, b) if conv == 1 else (b, c)
    G[1, 0, 1, 0] = G[0, 1, 0, 1] = stay
    G[0, 1, 1, 0] = G[1, 0, 0, 1] = swap
    return G

def apply_gate(psi, G, i, j, N):
    psi = np.tensordot(G, psi, axes=([2, 3], [i, j]))  # new axes 0,1 = sites i,j
    return np.moveaxis(psi, [0, 1], [i, j])

def transfer(N, K, mu, twist, conv):
    c = 2 * np.sin(np.pi / (4 * K))
    GA = gate(1 + mu, 1 - mu, c, conv); GB = gate(1 - mu, 1 + mu, c, conv)
    GBt = GB.copy()
    if twist:  # pi twist on the seam: charge crossing the bond (N-1,0) picks up -1
        GBt[0, 1, 1, 0] *= -1; GBt[1, 0, 0, 1] *= -1
    def T(v):
        psi = v.reshape((2,) * N)
        for i in range(0, N, 2):
            psi = apply_gate(psi, GA, i, i + 1, N)
        for i in range(1, N, 2):
            j = (i + 1) % N
            psi = apply_gate(psi, GBt if j == 0 else GB, i, j, N)
        return psi.reshape(-1)
    return T

def leading(N, K, mu, twist, conv, mag, cpar):
    idx = np.arange(2 ** N)
    pop = np.zeros(2 ** N, dtype=np.int64)
    for k in range(N):
        pop += (idx >> k) & 1
    mask = (pop == N // 2 + mag).astype(float)
    T = transfer(N, K, mu, twist, conv)
    def P(v):
        v = v * mask
        if cpar is not None:
            v = 0.5 * (v + cpar * v[::-1])
        return v
    op = LinearOperator((2 ** N, 2 ** N), matvec=lambda v: P(T(P(v))), dtype=float)
    v0 = P(np.random.default_rng(1).standard_normal(2 ** N))
    w = eigs(op, k=1, which="LM", v0=v0, tol=1e-13, ncv=20)[0]
    return abs(w[0])

def run(K, N, mu, conv):
    l0 = leading(N, K, mu, False, conv, 0, +1)
    l1 = leading(N, K, mu, False, conv, 1, None)
    lp = leading(N, K, mu, True, conv, 0, +1)
    lm = leading(N, K, mu, True, conv, 0, -1)
    s = N / (4 * np.pi)
    x01 = s * np.log(l0 / l1); xp = s * np.log(l0 / lp); xm = s * np.log(l0 / lm)
    dpi = xm - xp
    return dict(K=K, N=N, L=N / np.sqrt(2), mu=mu, conv=conv, x01=x01, x01_cft=1 / (4 * K),
                xpi_even=xp, xpi_odd=xm, xpi_cft=K / 4, delta_pi=dpi, g=-dpi / (2 * np.sqrt(2) * np.pi))

if __name__ == "__main__":
    K = float(sys.argv[1]); Ns = [int(s) for s in sys.argv[2].split(",")]
    mus = [float(s) for s in sys.argv[3].split(",")]; conv = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    out = sys.argv[5] if len(sys.argv) > 5 else None
    rows = []
    for N in Ns:
        for mu in mus:
            r = run(K, N, mu, conv); rows.append(r)
            print(" ".join(f"{k}={v:.8g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()), flush=True)
    if out:
        new = not os.path.exists(out)
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            new and w.writeheader(); w.writerows(rows)
