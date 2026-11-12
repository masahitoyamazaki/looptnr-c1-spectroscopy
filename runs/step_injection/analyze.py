"""Additivity of per-step injections of the dimerization seed (signed pi-twist readout, paper 1b).

g_pi(n) = -split_pi(n) / (2 sqrt2 pi), read at n = 16, 20, 24 (out/*.csv from si_run.jl).
Delta_S = g_pi(variant, steps S) - g_pi(reference).  Additivity: Delta_{5-7}, Delta_{8-11}, Delta_{5-11}
compared with the sums of the single-step Delta_m.
"""
import csv, glob, math, os
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")

def read(name):
    p = os.path.join(OUT, name + ".csv")
    if not os.path.exists(p): return None
    g = {}
    for r in csv.DictReader(l for l in open(p) if not l.startswith("#")):
        sp = float(r["split_pi"])
        if not math.isnan(sp): g[int(r["n"])] = -sp / (2 * math.sqrt(2) * math.pi)
    return g

for K, chi in ((1.5, 40), (1.5, 32)):
    ref = read(f"ref_K{K}_chi{chi}")
    if not ref or 24 not in ref:
        print(K, chi, "reference incomplete"); continue
    print(f"\nK={K} chi={chi}: g_pi(24) reference = {ref[24]:+.4e}")
    for v in ("chi64", "s80"):
        d = {}
        for m in range(5, 12):
            g = read(f"{v}_K{K}_chi{chi}_n{m}")
            if g and 24 in g: d[m] = g[24] - ref[24]
        print(f"  {v}: single-step Delta_m / |g_ref|: " + " ".join(f"{m}:{d[m]/abs(ref[24]):+.3f}" for m in sorted(d)))
        for name, S in (("n5-7", range(5, 8)), ("n8-11", range(8, 12)), ("n5-11", range(5, 12))):
            g = read(f"{v}_K{K}_chi{chi}_{name}")
            if g and 24 in g and all(m in d for m in S):
                tot = g[24] - ref[24]; s = sum(d[m] for m in S)
                print(f"    {name:6s}: Delta = {tot/abs(ref[24]):+.3f} |g_ref|,  sum of singles = {s/abs(ref[24]):+.3f} |g_ref|,  ratio {tot/s if s else float('nan'):.3f}")
