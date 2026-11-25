"""Referee round 4, major 3: is the seed complete at step 12?
(i) Calibration groups (twist campaign calibration.csv): response slope s(n)/r^(n/2) and the equivalent alternation
    mu_pred(n) = -delta_pi(n;0)/s(n) at eps/4, relative to its value at the last calibration step.
(ii) Integer chi scans (runs/early_xi/out): |g_pi(n)|/(sqrt2)^((2-K)n) at n=12,16,20 relative to n=24.
Usage: python seed_growth.py <calibration.csv> <early_xi/out>; in the data repository these are
external/fushio_twist_campaign/calibration.csv and runs/early_xi/out.  Output: seed_growth.csv"""
import csv, glob, math, os, sys, numpy as np
cal, ed = sys.argv[1], sys.argv[2]
R = list(csv.DictReader(open(cal))); out = []
for g in sorted({r["group"] for r in R}):
    rr = [r for r in R if r["group"] == g]
    eps = min(float(r["epsilon"]) for r in rr)
    rr = sorted([r for r in rr if float(r["epsilon"]) == eps], key=lambda r: int(r["step"]))
    ns = [int(r["step"]) for r in rr if int(r["step"]) >= 8]
    mp = {int(r["step"]): float(r["mu_pred"]) for r in rr}; sl = {int(r["step"]): float(r["rescaled_slope"]) for r in rr}
    nl = ns[-1]
    for n in ns:
        out.append(dict(set="calibration", group=g, n=n, mu_pred_rel=mp[n]/mp[nl], slope_rel=sl[n]/sl[nl]))
    print(g, "n_last", nl, "mu_pred(n)/mu_pred(last):", " ".join(f"{n}:{mp[n]/mp[nl]:.3f}" for n in ns),
          "| slope rel. range %.4f-%.4f" % (min(sl[n]/sl[nl] for n in ns), max(sl[n]/sl[nl] for n in ns)))
for K in (1.0, 1.5):
    rat = {12: [], 16: [], 20: []}
    for f in glob.glob(os.path.join(ed, f"E_K{K}_chi*.csv")):
        d = {int(r["n"]): r for r in csv.DictReader(l for l in open(f) if not l.startswith("#"))}
        h = lambda n: abs(float(d[n]["split_pi"])) / math.sqrt(2) ** ((2 - K) * n)
        for n in rat:
            rat[n].append(h(n) / h(24))
    for n in rat:
        v = np.array(rat[n]); out.append(dict(set="early_xi", group=f"K{K}", n=n, mu_pred_rel=float(np.median(v)), slope_rel=""))
    print(f"early_xi K={K}: median |g_pi(n)|/growth relative to n=24:", " ".join(f"{n}:{np.median(v):.3f} [{np.percentile(v,10):.2f},{np.percentile(v,90):.2f}]" for n, v in ((n, np.array(rat[n])) for n in rat)))
with open("seed_growth.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
