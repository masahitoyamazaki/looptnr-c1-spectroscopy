"""Three readouts of the dimerization coupling of the same uncompensated O(2) runs (2026-11-12, Yamazaki).
  (1) twisted: signed estimator g_pi = -delta_pi/(2 sqrt2 pi) of the pi-twisted tube (circumference 2 a_n)
      from Fushio/docs/six_vertex/dimerization_injection/support/twist_campaign_20261109/twist.csv (h = 0),
      rescaled to the standard tube (circumference 2 sqrt2 a_n) by (sqrt2)^(2-K);
  (2) standard: |g_d(n)| = sqrt(g2_12) * 2^((2-K)(n-12)/2) with the protocol of the merged paper 1b
      (rate fixed, |X| <= 0.02), from Otsuki/sixvertex/data/sixvertex_sg_xi_predict.csv;
  (3) correlation length: |g_d(n)| = sqrt2 kappa(p) [l(n)/(2 pi xi)]^(2-K), inverting the sine-Gordon relation
      with the measured xi of the same csv.
Compared at step N (twisted values exist every 4 steps).  Output: gd_three_way.csv
"""
import csv, math, sys
from pathlib import Path
TNR = Path(__file__).resolve().parents[3]
TW = TNR / "Fushio/docs/six_vertex/dimerization_injection/support/twist_campaign_20261109/twist.csv"
PR = TNR / "Otsuki/sixvertex/data/sixvertex_sg_xi_predict.csv"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24

def kappa(p):
    return math.gamma(p / (p + 1)) / (math.pi * math.gamma(1 / (p + 1))) * \
        (math.sqrt(math.pi) * math.gamma((p + 1) / 2) / (2 * math.gamma(p / 2))) ** (2 / (p + 1))
ell = lambda n: 2 * math.sqrt(2) * math.sqrt(2) ** n

tw = {}
for r in csv.DictReader(open(TW)):
    if float(r["h"]) == 0 and int(r["step"]) == N:
        tw[r["group"]] = float(r["g"])
pr = [r for r in csv.DictReader(open(PR)) if r["cut"] == "0.02" and r["rate"] == "fixed" and float(r["mu"] or 0) == 0
      and r["source"] in ("Yamazaki chi scan", "Yamazaki B9") and r["settled"] == "1"]
out = []
for grp, gtw in sorted(tw.items()):
    K = float(grp.split("_")[0][1:]); chi = int(grp.split("chi")[1])
    rows = [r for r in pr if abs(float(r["K"]) - K) < 1e-9 and int(r["chi"]) == chi]
    if not rows:
        continue
    r = sorted(rows, key=lambda r: r["source"] != "Yamazaki chi scan")[0]
    g_tw = gtw * math.sqrt(2) ** (2 - K)
    g_std = math.sqrt(float(r["g2_12"])) * 2 ** ((2 - K) * (N - 12) / 2)
    xi = float(r["xi_meas"]); p = K / (2 - K)
    g_xi = math.sqrt(2) * kappa(p) * (ell(N) / (2 * math.pi * xi)) ** (2 - K)
    out.append(dict(K=K, chi=chi, step=N, g_twist=g_tw, g_std=g_std, g_xi=g_xi, twist_over_std=abs(g_tw) / g_std,
                    xi_over_std=g_xi / g_std, xi_meas=xi, source=r["source"]))
w = csv.DictWriter(open(Path(__file__).with_name(f"gd_three_way_n{N}.csv"), "w"), fieldnames=list(out[0]))
w.writeheader(); [w.writerow(o) for o in out]
for o in out:
    print(f"K={o['K']:.2f} chi={o['chi']}: twist {o['g_twist']:+.3e}  std {o['g_std']:.3e}  xi {o['g_xi']:.3e}   |twist|/std={o['twist_over_std']:.3f}  xi/std={o['xi_over_std']:.3f}")
