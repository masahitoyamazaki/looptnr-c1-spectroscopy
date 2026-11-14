"""Ratio of the seed g_d^2(12) read with the protocol of Paper II (rate fixed to 2^(2-K), window n>=12, |X|<=0.02)
to the seed of the main O(2) fits of Paper I (runs/main/analysis/o2_fits.csv), for the unperturbed runs (mu=0).
Paper I, Sec. III A: the factor is 0.7-1.6 for K<=1.5 and about 4 at K=1.75, and varies by less than 20% with chi.
Input of the fixed-rate seeds: sixvertex_sg_xi_predict.csv of the collaboration repository (argv[1]); its values used here
are copied to protocol_factor.csv (column g2_12_fixed).
Output: protocol_factor.csv
"""
import csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "..", "main", "analysis", "o2_fits.csv")
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "..", "TNR", "Otsuki", "sixvertex", "data", "sixvertex_sg_xi_predict.csv")
main = {(round(float(r["K"]), 3), int(r["chi"])): float(r["gd2_n12"])
        for r in csv.DictReader(open(MAIN)) if r["gd2_n12"] not in ("", "nan")}
rows = []
for r in csv.DictReader(open(SRC)):
    if r["cut"] == "0.02" and r["rate"] == "fixed" and float(r["mu"]) == 0 and r["g2_12"] \
            and r["source"] in ("saved", "Yamazaki B2", "Yamazaki B9"):
        k = (round(float(r["K"]), 3), int(r["chi"]))
        if k in main:
            rows.append({"K": k[0], "chi": k[1], "source": r["source"], "g2_12_fixed": r["g2_12"],
                         "g2_12_main": main[k], "factor": float(r["g2_12"]) / main[k]})
rows.sort(key=lambda d: (d["K"], d["chi"], d["source"]))
with open(os.path.join(HERE, "protocol_factor.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for d in rows:
    print(f'K={d["K"]:<5} chi={d["chi"]:<3} {d["source"]:<12} factor={d["factor"]:.3f}')
