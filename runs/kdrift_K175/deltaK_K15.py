"""Shift dK of the Luttinger parameter that would remove the shortfall of xi_pred/xi at K = 1.5 with the cut |X| <= 0.1
(Paper 1b, App. D, "Bias of the window cut"; Otsuki R-34; first computed in Otsuki/sixvertex/sixvertex_sg_xi.md, 4993eae3).

K is replaced by K + dK in Eq. (xi) (exponent 1/(2-K) and kappa_SG(p), p = K/(2-K)) and in 4K alpha_{0,1}(K) of Eq. (g2),
with the amplitude A of the growing term at the window end n_end, and the tube length L(n_end), held fixed:
    xi(K) = L/(2 pi) [sqrt2 kappa_SG(p) / |g|]^(1/(2-K)),   g^2 = |A| / (4K alpha_{0,1}(K)).
dK solves xi(K + dK) / xi(K) = 1 / ratio, ratio = xi_pred/xi measured.
(a) per run: |A| from the run's own seed (g^2 at n_end recovered from xi_pred_end), ratio from the same row;
(b) at fixed |X(n_end)| = |A| = 0.02, 0.05, 0.1 for the range of ratios.
alpha_{0,1}(K) is the column m1 of sixvertex_g2_alpha.csv (step 0.05 in K), interpolated with a cubic spline
(np.interp, as in kdrift_all.py, gives the same dK to within 2%: printed as a check).
Runs: as Fig. 2(a) of the paper (Yamazaki/runs/xi_sg_20261112/scripts/common.py, runs(rows, 1.5, 0.1)):
settled runs, rate fixed to 2^(2-K), saved checkpoints excluded, duplicates merged.
Usage: python deltaK_K15.py [root of the collaboration repository].  Output: deltaK_K15.csv, deltaK_K15.out (stdout)."""
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
T = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[2]
D = T / "Otsuki" / "sixvertex" / "data"
K0, CUT = 1.5, 0.1

al = {float(r["K"]): float(r["m1"]) for r in csv.DictReader(open(D / "sixvertex_g2_alpha.csv"))}   # K = 1.5 appears twice
al = sorted((k, a) for k, a in al.items() if abs(k * 20 - round(k * 20)) < 1e-9)   # the 0.05 grid (drops K = 0.6267)
Kg, ag = np.array([k for k, _ in al]), np.array([a for _, a in al])
SPL = CubicSpline(Kg, ag)
alpha_spline = lambda K: float(SPL(K))
alpha_linear = lambda K: float(np.interp(K, Kg, ag))


def kappa_sg(K):
    p = K / (2 - K)
    return (math.gamma(p / (p + 1)) / math.gamma(1 / (p + 1)) / math.pi
            * (math.sqrt(math.pi) * math.gamma((p + 1) / 2) / (2 * math.gamma(p / 2))) ** (2 / (p + 1)))


def ell(n):
    return 2 * math.sqrt(2) * math.sqrt(2) ** n


def ln_xi(K, A, n, alpha):
    """ln xi of Eq. (xi) with g^2 = |A| / (4 K alpha(K)) at step n"""
    g = math.sqrt(A / (4 * K * alpha(K)))
    return math.log(ell(n) / (2 * math.pi)) + math.log(math.sqrt(2) * kappa_sg(K) / g) / (2 - K)


def dK_needed(A, n, ratio, alpha):
    f = lambda d: ln_xi(K0 + d, A, n, alpha) - ln_xi(K0, A, n, alpha) + math.log(ratio)
    return brentq(f, -0.2, 0.2)


def dlnxi_dK(A, n, alpha, h=1e-5):
    return (ln_xi(K0 + h, A, n, alpha) - ln_xi(K0 - h, A, n, alpha)) / (2 * h)


def num(v):
    try:
        return float(v)
    except ValueError:
        return v


rows = [{k: num(v) for k, v in r.items()} for r in csv.DictReader(open(D / "sixvertex_sg_xi_predict.csv"))]
sel, seen = [], set()
for r in rows:
    if not (r["K"] == K0 and r["cut"] == CUT and r["rate"] == "fixed" and r["settled"] == 1 and r["source"] != "saved"
            and isinstance(r["ratio_end"], float) and r["ratio_end"] == r["ratio_end"]):
        continue
    key = (r["chi"], round(r["xi_meas"], 3))
    if key in seen:
        continue
    seen.add(key)
    sel.append(r)

print(f"K = {K0}, cut |X| <= {CUT}, rate fixed: {len(sel)} runs (as in Fig. 2(a))")
print(f"alpha_01(1.5) = {alpha_spline(K0):.5f}, d ln alpha_01/dK = {float(SPL(K0, 1)) / alpha_spline(K0):.4f} (spline)")
out = []
for r in sorted(sel, key=lambda r: (r["chi"], r["mu"])):
    ne = int(r["window_end"])
    g = math.sqrt(2) * kappa_sg(K0) * (ell(ne) / (2 * math.pi * r["xi_pred_end"])) ** (2 - K0)   # |g_d(n_end)|
    A = 4 * K0 * alpha_spline(K0) * g * g          # |A|: growing term of Eq. (fit) at n_end
    d = dK_needed(A, ne, r["ratio_end"], alpha_spline)
    dl = dK_needed(A, ne, r["ratio_end"], alpha_linear)
    out.append(dict(run=r["run"], chi=int(r["chi"]), mu=r["mu"], n_end=ne, A_end=A, ratio=r["ratio_end"],
                    dlnxi_dK=dlnxi_dK(A, ne, alpha_spline), dK=d, dK_linear_alpha=dl))
with open(HERE / "deltaK_K15.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)

print("\n(a) per run: |A(n_end)| from the run's seed, ratio = xi_pred/xi of the same run")
print(f"{'chi':>4} {'mu':>10} {'n_end':>5} {'|A|':>7} {'ratio':>6} {'dlnxi/dK':>9} {'dK':>8} {'dK(lin)':>8}")
for o in out:
    print(f"{o['chi']:>4} {o['mu']:>10.3g} {o['n_end']:>5} {o['A_end']:>7.4f} {o['ratio']:>6.3f} {o['dlnxi_dK']:>9.2f} "
          f"{o['dK']:>+8.4f} {o['dK_linear_alpha']:>+8.4f}")
for lab, oo in (("all", out), ("chi >= 24", [o for o in out if o["chi"] >= 24])):
    dk = np.array([o["dK"] for o in oo]); ra = np.array([o["ratio"] for o in oo]); aa = np.array([o["A_end"] for o in oo])
    print(f"  {lab}: {len(oo)} runs, ratio {ra.min():.3f}-{ra.max():.3f}, |A(n_end)| {aa.min():.3f}-{aa.max():.3f}, "
          f"dK {dk.min():+.4f} to {dk.max():+.4f}, median {np.median(dk):+.4f}")

print("\n(b) fixed |X(n_end)| = |A|, ratio over the range of (a) (chi >= 24)")
oo = [o for o in out if o["chi"] >= 24]
rmin, rmax = min(o["ratio"] for o in oo), max(o["ratio"] for o in oo)
n_ref = int(np.median([o["n_end"] for o in oo]))   # dK does not depend on n (L cancels in the ratio); printed for the record
for A in (0.02, 0.05, 0.1):
    print(f"  |X| = {A:<5}: dln xi/dK = {dlnxi_dK(A, n_ref, alpha_spline):6.2f};  dK = {dK_needed(A, n_ref, rmax, alpha_spline):+.4f} "
          f"(ratio {rmax:.3f}) to {dK_needed(A, n_ref, rmin, alpha_spline):+.4f} (ratio {rmin:.3f})")
print("\nSign: dK > 0 is needed (xi_pred too short and dln xi/dK > 0 at fixed |A|); the drift of Paper 1a at the end of the")
print("|X| <= 0.1 window is dK ~ -0.04 at K = 1.5 (Otsuki/sixvertex/sixvertex_sg_xi.md), opposite in sign.")
