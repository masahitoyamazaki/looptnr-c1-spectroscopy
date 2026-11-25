"""Paper 1b, referee round 5 (majors 1, 2): all groups of the pre-registered twist campaign and the source fit.
Inputs (argv[1] = campaign folder; in the data repository external/fushio_twist_campaign/):
  progress.md (n_e of the mu=0 and mu_pred runs), predictions/<group>.json (frozen mu_pred = input_h, status),
  calibration.csv (mu_pred(n) = -delta_pi(n;0)/s(n) at the smallest epsilon).
Source fit: for n >= 12, mu_pred(n) = a + c r^{-n}, r = (sqrt2)^{2-K}, the solution of Eq. (5) of Paper 1b with a
constant source in units of the first-order growth.  f12 = -c r^{-12}/a and fcal = -c r^{-ncal}/a are the fractions of
the asymptotic amplitude a injected after n = 12 and after n_cal; delay_src = 2 ln(1/|fcal|)/ln 2^{2-K} is the delay
expected if the frozen counterterm cancelled the amplitude present at n_cal exactly.
Output: campaign_summary.csv"""
import csv, json, math, os, re, sys, numpy as np
C = sys.argv[1]
ne = {}
for l in open(os.path.join(C, "progress.md")):
    m = re.match(r"\| (K[\d.]+_chi\d+)_(zero|predicted) \| [^|]+\| \d+ \| \d+ \| (\d+) \|", l)
    if m:
        ne[(m.group(1), m.group(2))] = int(m.group(3))
cal = list(csv.DictReader(open(os.path.join(C, "calibration.csv"))))
rows = []
for g in sorted({r["group"] for r in cal}, key=lambda g: (float(g[1:].split("_")[0]), int(g.split("chi")[1]))):
    K = float(g[1:].split("_")[0]); chi = int(g.split("chi")[1])
    rr = [r for r in cal if r["group"] == g]
    eps = min(float(r["epsilon"]) for r in rr)
    y = {int(r["step"]): float(r["mu_pred"]) for r in rr if float(r["epsilon"]) == eps}
    pf = os.path.join(C, "predictions", f"{g}.json")
    pr = json.load(open(pf)) if os.path.exists(pf) else {}
    ncal = pr.get("calibration_step", max(y))
    ns = np.array([n for n in sorted(y) if n >= 12]); r = math.sqrt(2) ** (2 - K)
    fit = dict(a=np.nan, f12=np.nan, fcal=np.nan, fit_rms=np.nan, delay_src=np.nan)
    if len(ns) >= 2:
        A = np.vstack([np.ones(len(ns)), r ** (-ns.astype(float))]).T; yy = np.array([y[n] for n in ns])
        (a, c), *_ = np.linalg.lstsq(A, yy, rcond=None)
        res = yy - A @ np.array([a, c])
        f12, fcal = -c * r ** -12 / a, -c * r ** (-float(ncal)) / a
        fit = dict(a=a, f12=f12, fcal=fcal, fit_rms=float(np.sqrt(np.mean((res / a) ** 2))) if len(ns) > 2 else np.nan,
                   delay_src=2 * math.log(1 / abs(fcal)) / math.log(2 ** (2 - K)) if fcal != 0 else np.nan)
    sign_change = any(y[n] * y[max(y)] < 0 for n in y if n >= 8)
    n0, n1 = ne.get((g, "zero")), ne.get((g, "predicted"))
    rows.append(dict(group=g, K=K, chi=chi, n_cal=ncal, status=pr.get("status", "no calibration"),
                     mu_pred=pr.get("input_h", np.nan), mu_pred_last=y[max(y)], sign_change_after_8=sign_change,
                     ne_zero=n0, ne_pred=n1, delay=(n1 - n0) if (n0 is not None and n1 is not None) else None, **fit))
with open("campaign_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for q in rows:
    print(f"{q['group']:12s} ncal={q['n_cal']} {q['status']:14s} mu_pred={q['mu_pred']:+.3e} n_e {q['ne_zero']}->{q['ne_pred']} "
          f"delay={q['delay']} | f12={q['f12']:+.3f} fcal={q['fcal']:+.4f} rms={q['fit_rms']:.3f} delay_src={q['delay_src']:.1f} signchg={q['sign_change_after_8']}")
