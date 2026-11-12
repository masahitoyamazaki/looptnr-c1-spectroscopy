"""Table of the two independent analyses (Otsuki: |X|<=0.02, last two steps; Yamamoto: |X|<=0.05, median of the
last three converged steps) on the common chi scan, written to tables/crosscheck.tex.
Usage (from paper2_draft/): ~/miniforge3/bin/python scripts/crosscheck_table.py
"""
import math

from common import PAPER, YAM, predict_rows, read


def sci(x, d=2):
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    return f"${m:.{d}f}\\times10^{{{e}}}$"


rows = predict_rows()
Y = {(r["K"], int(r["chi"])): r for r in read(YAM / "comparison.csv")}
lines = []
for (K, chi), y in sorted(Y.items()):
    o = [r for r in rows if r["K"] == K and r["chi"] == chi and r["mu"] == 0 and r["source"] == "Yamazaki chi scan"
         and r["cut"] == 0.02 and r["rate"] == "fixed"]
    if not o:
        continue
    o = o[0]
    g = math.sqrt(o["g2_12"])
    lines.append(f"      {K:g} & {chi} & {sci(g)} & {sci(y['h12'])} & {sci(o['xi_meas'], 3)} & "
                 f"{o['xi_pred_end'] / o['xi_meas']:.3f} & {y['xi_pred'] / y['xi_measured']:.3f} \\\\")
(PAPER / "tables").mkdir(exist_ok=True)
(PAPER / "tables" / "crosscheck.tex").write_text("\n".join(lines) + "\n")
print(len(lines), "rows")
