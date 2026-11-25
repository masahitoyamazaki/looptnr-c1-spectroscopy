"""Print every number quoted in the text of paper 2 that is derived from the CSVs (provenance in README).
Usage: ~/miniforge3/bin/python scripts/paper_numbers.py > scripts/paper_numbers.out
"""
import math
import statistics

from common import KS, YAM, alpha01, kappa_K, predict_rows, read, runs

rows = predict_rows()
print("== runs per K (protocol |X|<=0.02, r fixed; duplicates merged)")
tot = 0
for K in KS:
    v = runs(rows, K)
    rr = sorted(r["ratio_end"] for r in v)
    no16 = sorted(r["ratio_end"] for r in v if r["chi"] != 16)
    tot += len(v)
    print(f"K={K}: N={len(v)}, median={statistics.median(rr):.4f}, range={rr[0]:.3f}-{rr[-1]:.3f}, "
          f"range chi!=16 = {no16[0]:.3f}-{no16[-1]:.3f}, xi_meas {min(r['xi_meas'] for r in v):.3g}-{max(r['xi_meas'] for r in v):.3g}")
    c = {}
    for r in v:
        c[r["source"]] = c.get(r["source"], 0) + 1
    print("   sources:", c)
print("total runs:", tot)

print("== cut 0.1, r fixed")
for K in KS:
    rr = sorted(r["ratio_end"] for r in runs(rows, K, 0.1))
    print(f"K={K}: {rr[0]:.3f}-{rr[-1]:.3f}")

print("== K=1.75: ratio vs end of window")
v = runs(rows, 1.75)
short = [r for r in v if r["window_end"] <= 69]
long_ = [r for r in v if r["window_end"] >= 80]
print(f"ends <=69: {len(short)} runs, mean {statistics.mean(r['ratio_end'] for r in short):.3f}, ends "
      f"{min(r['window_end'] for r in short):.0f}-{max(r['window_end'] for r in short):.0f}")
print(f"ends >=80: {len(long_)} runs, mean {statistics.mean(r['ratio_end'] for r in long_):.3f}, ends "
      f"{min(r['window_end'] for r in long_):.0f}-{max(r['window_end'] for r in long_):.0f}")

print("== repeated runs of the same condition (different batches)")
for K in KS:
    sel = [r for r in rows if r["K"] == K and r["cut"] == 0.02 and r["rate"] == "fixed" and r["settled"] == 1
           and r["source"] != "saved" and r["mu"] == 0]
    by = {}
    for r in sel:
        by.setdefault(r["chi"], []).append(r["xi_meas"])
    for chi, xs in sorted(by.items()):
        xs = sorted(set(round(x, 3) for x in xs))
        if len(xs) > 1:
            print(f"K={K} chi={chi:.0f}: xi = {xs}, spread {xs[-1] / xs[0] - 1:.2e}")

print("== L/xi at n_e (cut 0.1, r free) and second-order estimate")
# The column L_over_xi_at_n_e of the Otsuki table is taken at the first step with |X| > 0.1, i.e., at n_e + 1;
# divide by sqrt(2) for n_e, the last step with |X| <= 0.1 (Fushio 1b R-05).
for K in KS:
    ne = sorted(r["L_over_xi_at_n_e"] / math.sqrt(2) for r in runs(rows, K, 0.1, "free")
                if isinstance(r["L_over_xi_at_n_e"], float) and r["L_over_xi_at_n_e"] == r["L_over_xi_at_n_e"])
    est = 2 * math.pi * (math.sqrt(0.1 / (4 * K * alpha01(K))) / (math.sqrt(2) * kappa_K(K))) ** (1 / (2 - K))
    print(f"K={K}: {ne[0]:.2f}-{ne[-1]:.2f}  estimate {est:.2f}")

print("== saved runs K=0.6, 0.9")
for r in rows:
    if r["source"] == "saved" and r["cut"] == 0.02 and r["rate"] == "fixed":
        print(r["run"], f"{r['ratio_end']:.4f}", f"xi={r['xi_meas']:.6g}")

print("== cross-check with Yamamoto (common chi scan)")
Y = {(r["K"], int(r["chi"])): r for r in read(YAM / "comparison.csv")}
dg, dp, dm = [], [], []
for (K, chi), y in sorted(Y.items()):
    o = [r for r in rows if r["K"] == K and r["chi"] == chi and r["mu"] == 0 and r["source"] == "Yamazaki chi scan"
         and r["cut"] == 0.02 and r["rate"] == "fixed"]
    if not o:
        continue
    o = o[0]
    g = math.sqrt(o["g2_12"])
    dg.append(abs(g / y["h12"] - 1))
    dp.append(abs(o["xi_pred_end"] / y["xi_pred"] - 1))
    dm.append(abs(o["xi_meas"] / y["xi_measured"] - 1))
print(f"{len(dg)} common runs: max |g/h-1| = {max(dg):.4f}, max |pred_O/pred_Y-1| = {max(dp):.4f}, "
      f"max |meas_O/meas_Y-1| = {max(dm):.2e}")
print("Yamamoto only, K=0.6 (B8), measured/predicted:",
      [f"{Y[(0.6, c)]['measured_over_pred']:.4f}" for c in (32, 40, 48, 64)])
print("Yamamoto K=0.5:", [f"{Y[(0.5, c)]['measured_over_pred']:.4f}" for c in (32, 40, 48, 64)],
      "no-log:", [f"{Y[(0.5, c)]['xi_measured'] / Y[(0.5, c)]['xi_no_log']:.3f}" for c in (32, 40, 48, 64)])

print("== chi != 16 variants")
for K in KS:
    rr = sorted(r["ratio_end"] for r in runs(rows, K, 0.1) if r["chi"] != 16)
    print(f"cut 0.1, K={K}: {rr[0]:.3f}-{rr[-1]:.3f}  (chi=16 runs: {[round(r['ratio_end'],3) for r in runs(rows, K, 0.1) if r['chi'] == 16]})")
v = [r for r in runs(rows, 1.75) if r["chi"] != 16]
short = [r for r in v if r["window_end"] <= 69]
print(f"K=1.75 chi!=16 ends<=69: {len(short)} runs mean {statistics.mean(r['ratio_end'] for r in short):.3f} "
      f"ends {min(r['window_end'] for r in short):.0f}-{max(r['window_end'] for r in short):.0f}")
print("K=1.75 runs:", sorted((r['chi'], r['mu'], r['window_end'], round(r['ratio_end'], 3)) for r in runs(rows, 1.75)))
