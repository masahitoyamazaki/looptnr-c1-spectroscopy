"""Bond-dimension history of the first RG steps (Paper 1b, Sec. VII; referee: first truncated step vs chi).
For each run, the step at which the bond dimension is first capped by chi and the loop optimization runs
(sweeps > 0) is reported, with dims_A of steps 0-7 from steps.csv(.gz).
Output: first_truncation.csv
"""
import csv, glob, gzip, os
HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "..", "main", "data")
if not os.path.isdir(MAIN):
    MAIN = os.path.join(HERE, "..", "main", "data")
ADD = os.path.join(HERE, "..", "additional", "data")
if not os.path.isdir(ADD):
    ADD = os.path.join(HERE, "..", "additional", "data")
runs = sorted(glob.glob(os.path.join(MAIN, "CU1_K*_chi*"))) + sorted(glob.glob(os.path.join(ADD, "B5_K*_chi96")))
out = []
for d in runs:
    p = os.path.join(d, "steps.csv")
    f = open(p) if os.path.exists(p) else gzip.open(p + ".gz", "rt")
    rows = list(csv.DictReader(f))[:8]
    dims = [int(r["dims_A"].split(";")[0]) for r in rows]; sw = [int(r["sweeps"]) for r in rows]
    first = next((int(r["step"]) for r in rows if int(r["sweeps"]) > 0), None)
    name = os.path.basename(d)
    out.append(dict(run=name, first_optimized_step=first, dims_0_7=" ".join(map(str, dims)), sweeps_0_7=" ".join(map(str, sw))))
    print(f"{name:22s} first step with loop optimization: {first}   dims {dims}  sweeps {sw}")
with open(os.path.join(HERE, "first_truncation.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
