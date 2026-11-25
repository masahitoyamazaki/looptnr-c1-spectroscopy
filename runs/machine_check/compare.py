"""Compare the two machines (out/ximachine_mac.csv, out/ximachine_a6000.csv): seed, predicted xi, and the xi ratio
from the steps at which |X| crosses 0.1, 0.3, 1, and 2 (xi ratio = 2^(dn/2)). Usage: python compare.py"""
import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = {m: {int(r["n"]): {k: float(v) for k, v in r.items()} for r in csv.DictReader(open(HERE / "out" / f"ximachine_{m}.csv"))}
     for m in ("mac", "a6000")}
for n in (8, 12, 16, 20, 24):
    a, b = D["mac"][n], D["a6000"][n]
    print(f"n={n}: g_d^pi mac {a['gd_pi']:+.4e} a6000 {b['gd_pi']:+.4e} ratio {b['gd_pi'] / a['gd_pi']:.3f}; "
          f"predicted xi ratio {b['xi_pred'] / a['xi_pred']:.3f}")


def cross(d, thr):
    ns = sorted(d)
    for n0, n1 in zip(ns, ns[1:]):
        x0, x1 = abs(d[n0]["X"]), abs(d[n1]["X"])
        if x0 < thr <= x1:
            return n0 + (math.log(thr) - math.log(x0)) / (math.log(x1) - math.log(x0))


for thr in (0.1, 0.3, 1.0, 2.0):
    cm, ca = cross(D["mac"], thr), cross(D["a6000"], thr)
    print(f"|X|={thr}: n mac {cm:.3f} a6000 {ca:.3f} -> measured xi ratio {2 ** ((ca - cm) / 2):.3f}")
for m in D:
    print(m, "n_e", max(n for n in D[m] if abs(D[m][n]["X"]) <= 0.1))
