"""DMRG extension of xxz_alpha_check.py to longer periodic chains (2026-11-06).

Same observable as the exact diagonalization: periodic XXZ chain with bond alternation 1 +- delta,
Delta = -cos(pi / (2K)); X = 1 - 4K x_{0,1} (ground state of S^z = 1), X_J = 1 - x_J with x_J the average of the
first two excited states of S^z = 0 (the current pair J, Jbar for K > 1: e = +-1 lie at x = K > 1); the delta^2
coefficient by Richardson extrapolation from delta and 2 delta; dX_i / delta^2 = a_i + b_i L^{2(2-K)};
b_J / b_X = alpha_J / (4 K alpha_{0,1}).  Uses TeNPy (finite MPS with periodic couplings, S^z conserved).
Usage:  python3 xxz_alpha_dmrg.py K L [L ...]        (prints one line per L; collect with --fit K L1 L2 ...)
"""
import sys, math, json, os
import numpy as np
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "xxz_alpha_dmrg.json")
ALPHA = {0.75: (40.640, 47.840), 1.0: (27.364, 39.478), 1.25: (20.320, 32.578), 1.5: (17.089, 27.059)}


class DimerXXZ(CouplingMPOModel):
    def init_sites(self, p):
        return SpinHalfSite(conserve="Sz")

    def init_terms(self, p):
        L, Delta, delta = p.get("L", 16), p.get("Delta", 0.0), p.get("delta", 0.0)
        w = np.array([1 + (-1) ** j * delta for j in range(L)])   # bond j-(j+1), periodic
        self.add_coupling(0.5 * w, 0, "Sp", 0, "Sm", 1, plus_hc=True)
        self.add_coupling(Delta * w, 0, "Sz", 0, "Sz", 1)


def energies(L, Delta, delta, chi):
    M = DimerXXZ(dict(L=L, Delta=Delta, delta=delta, bc_MPS="finite", bc_x="periodic"))
    opts = dict(mixer=True, max_E_err=1e-12, max_S_err=1e-8, max_sweeps=40, min_sweeps=6,
                trunc_params=dict(chi_max=chi, svd_min=1e-12), combine=True)
    up_dn = ["up", "down"] * (L // 2)
    psi0 = MPS.from_product_state(M.lat.mps_sites(), up_dn, bc="finite")
    E0, psi_g = dmrg.run(psi0, M, opts)["E"], psi0
    ex = []
    orth = [psi_g]
    for _ in range(2):
        p = MPS.from_product_state(M.lat.mps_sites(), up_dn, bc="finite")
        p.perturb(dict(N_steps=6, trunc_params=dict(chi_max=8)), close_1=True, canonicalize=True)
        ex.append(dmrg.run(p, M, opts, orthogonal_to=list(orth))["E"]); orth.append(p)
    st1 = ["up", "down"] * (L // 2); st1[1] = "up"                  # S^z = 1
    p1 = MPS.from_product_state(M.lat.mps_sites(), st1, bc="finite")
    E1 = dmrg.run(p1, M, opts)["E"]
    return E0, E1, 0.5 * (ex[0] + ex[1])


def run(K, L, chi, d=5e-3):
    Delta = -math.cos(math.pi / (2 * K)); gam = math.acos(Delta); v = math.pi * math.sin(gam) / (2 * gam)
    sc = L / (2 * math.pi * v)
    out = {}
    for dd in (0.0, d, 2 * d):
        E0, E1, EJ = energies(L, Delta, dd, chi)
        out[dd] = (1 - 4 * K * (E1 - E0) * sc, 1 - (EJ - E0) * sc)
    res = []
    for i in (0, 1):
        f1 = out[d][i] - out[0.0][i]; f2 = out[2 * d][i] - out[0.0][i]
        res.append((16 * f1 - f2) / (12 * d * d))
    rec = dict(K=K, L=L, chi=chi, d=d, X0=out[0.0][0], XJ0=out[0.0][1], dX=res[0], dXJ=res[1])
    data = json.load(open(OUT)) if os.path.exists(OUT) else []
    data = [r for r in data if not (r["K"] == K and r["L"] == L and r["chi"] == chi)] + [rec]
    json.dump(data, open(OUT, "w"), indent=1)
    print(json.dumps(rec), flush=True)


def fit(K, Ls):
    data = {r["L"]: r for r in json.load(open(OUT)) if r["K"] == K and r["L"] in Ls}
    Ls = sorted(data); p = 2 * (2 - K)
    M = np.vstack([np.ones(len(Ls)), np.array(Ls, float) ** p]).T
    bX = np.linalg.lstsq(M, [data[L]["dX"] for L in Ls], rcond=None)[0][1]
    bJ = np.linalg.lstsq(M, [data[L]["dXJ"] for L in Ls], rcond=None)[0][1]
    a01, aJ = ALPHA[K]
    print(f"K={K} L={Ls}: b_J/b_X = {bJ / bX:.4f}; prediction {aJ / (4 * K * a01):.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "--fit":
        fit(float(sys.argv[2]), [int(x) for x in sys.argv[3:]])
    else:
        K = float(sys.argv[1]); chi = int(os.environ.get("CHI", "256"))
        for L in map(int, sys.argv[2:]):
            run(K, L, chi)
