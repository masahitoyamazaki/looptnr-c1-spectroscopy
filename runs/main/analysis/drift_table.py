"""Paper I, Table of the direct and induced shifts of K (Sec. IV F).

direct  |delta K| = |dK_n/dn| (floor window, drift_summary.csv) x n_e
induced |delta K| = pi^2 K^2 g_d^2(n_e) / (2 - K), with g_d^2(n_e) = -A / (4 K alpha_01) from the growth fit
                    (o2_fits.csv; A is the growth term at n_e) -- Eq. (flow) of the paper at fixed K.
Usage: python3 drift_table.py  (after analyze.py and drift.py); writes drift_table.csv
"""
import csv, math, os
HERE = os.path.dirname(os.path.abspath(__file__))
fits = {(float(r["K"]), int(r["chi"])): r for r in csv.DictReader(open(os.path.join(HERE, "o2_fits.csv")))}
drift = {(float(r["K"]), int(r["chi"])): r for r in csv.DictReader(open(os.path.join(HERE, "drift_summary.csv")))}
alpha = {round(float(r["K"]), 4): float(r["m1"]) for r in csv.DictReader(open(os.path.join(HERE, "sixvertex_g2_alpha.csv")))}
rows = []
for (K, chi), f in sorted(fits.items()):
    d = drift.get((K, chi))
    if not d or not d["dK_per_step"] or K >= 2 or K not in (0.75, 1.0, 1.25, 1.5, 1.75):
        continue
    if not f["n_e"] or not f["A"]:
        continue
    ne, A = int(f["n_e"]), float(f["A"])
    if not math.isfinite(A):
        continue
    gd2 = -A / (4 * K * alpha[round(K, 4)])
    direct = abs(float(d["dK_per_step"])) * ne
    induced = math.pi ** 2 * K ** 2 * gd2 / (2 - K)
    rows.append(dict(K=K, chi=chi, n_e=ne, dK_per_step=float(d["dK_per_step"]), direct=direct, induced=induced,
                     ratio=induced / direct))
with open(os.path.join(HERE, "drift_table.csv"), "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print("K=%.2f chi=%2d n_e=%2d direct=%.2e induced=%.3f ratio=%.1f" % (r["K"], r["chi"], r["n_e"], r["direct"], r["induced"], r["ratio"]))
