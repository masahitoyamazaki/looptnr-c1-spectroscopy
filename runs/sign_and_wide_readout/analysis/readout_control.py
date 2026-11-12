"""Readout control for the wide tube (followup C, 2026-11-10).

The wide tube applies one extra coarse-graining (MPO_opt/reduced_MPO) before the transfer-matrix diagonalization,
truncated to chi in followup_B W1 (wide_readout.jl).  Here the same checkpoints are re-read with that readout
truncation relaxed to 2 chi (../scripts/wide_readout_trunc.jl, CHI_READ=2chi) and analyzed with the same fit
(followup_B analysis/wide_tube.py).  If the 5^{2-K} factor and the amplitude ratio are geometric, they must not
change with CHI_READ.   Usage: python3 readout_control.py -> readout_control.csv
"""
import csv, gzip, math, os, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "additional", "analysis"))
import wide_tube as wt  # noqa: E402
from analyze import analyze_o2, ALPHA  # noqa: E402

orig = wt.wide_levels
def relaxed_levels(K, chi):
    lev = defaultdict(list)
    for r in csv.DictReader(gzip.open(os.path.join(HERE, "..", "data", "out_readout", f"K{K:.4f}_chi{chi}_read{2*chi}.csv.gz"), "rt")):
        lev[int(r["step"])].append(r)
    return dict(lev)

alpha = {round(float(r["K"]), 4): r for r in csv.DictReader(open(ALPHA))}
rows = []
for K in (1.0, 1.5):
    for chi in (32, 40):
        a = alpha[round(K, 4)]
        nar = analyze_o2(os.path.join(wt.BASE, f"CU1_K{K:.4f}_chi{chi}"), K, chi, a)
        res = {}
        for tag, fn in (("read_chi", orig), ("read_2chi", relaxed_levels)):
            wt.wide_levels = fn
            w = wt.fit_wide(K, chi, a)
            nc = min(nar["n_e"], w["n_e"])
            g_n = -nar["A"] * nar["r"] ** (nc - nar["n_e"]); g_w = -w["A"] * w["r"] ** (nc - w["n_e"])
            fixw = wt.fixed_rate_amplitude(K, chi, math.log(nar["r"])) / (nar["A"] * nar["r"] ** (-nar["n_e"]))
            res[tag] = dict(n_e=w["n_e"], r=w["r_over_pred"], q=w["ratio_over_pred"], g2r=g_w / g_n, fix=fixw)
        wt.wide_levels = orig
        # direct level comparison at common steps
        A, B = orig(K, chi), relaxed_levels(K, chi)
        def x01(rows): return min(float(r["x"]) for r in rows if "(1, 2)" in r["sector"])
        common = [n for n in sorted(B) if n in A and 12 <= n <= res["read_chi"]["n_e"]]
        dmax = max(abs(x01(A[n]) - x01(B[n])) for n in common)
        dX = max(abs(x01(A[n]) - x01(B[n])) / abs(1 / (4 * K) - x01(A[n])) for n in common)
        row = dict(K=K, chi=chi, pred=5 ** (2 - K), q_std=nar["ratio_over_pred"], max_dx01=dmax, max_rel_dX=dX)
        for tag, v in res.items():
            for k, x in v.items():
                row[f"{k}_{tag}"] = x
        rows.append(row)
        print(f"K={K} chi={chi}: n_e {res['read_chi']['n_e']}/{res['read_2chi']['n_e']}, r/2^(2-K) {res['read_chi']['r']:.4f}/{res['read_2chi']['r']:.4f}, "
              f"ratio {res['read_chi']['q']:.4f}/{res['read_2chi']['q']:.4f} (std {nar['ratio_over_pred']:.4f}), "
              f"g2 wide/std {res['read_chi']['g2r']:.3f}/{res['read_2chi']['g2r']:.3f}, same-rate {res['read_chi']['fix']:.3f}/{res['read_2chi']['fix']:.3f} "
              f"(pred {5**(2-K):.3f}); max|dx01|={dmax:.2e}, max rel change of X={dX:.2e}")
with open(os.path.join(HERE, "readout_control.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
