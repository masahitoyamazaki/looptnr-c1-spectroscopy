"""Machine check: X(n) of the new runs (a second workstation, out/) against the deposited spectra of the same (K, chi)
(sixvertex_sg_xi_levels.csv, Miyabi / earlier runs).  Prints the step at which the two first differ by
more than 1e-8, 1e-4 and 1e-2 in X, and the max |dX| for n <= 12 and n <= 24.  Output: machine_check.csv"""
import csv, glob, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "..", "Otsuki", "sixvertex", "data", "sixvertex_sg_xi_levels.csv")
dep = {}
for r in csv.DictReader(open(SRC)):
    if float(r["mu"]) == 0 and not r["run"].startswith(("ab", "saved")):
        dep.setdefault((float(r["K"]), int(r["chi"])), {})[int(r["n"])] = (float(r["X"]), r["run"])
out = []
for f in sorted(glob.glob(os.path.join(HERE, "out", "E_K*_chi*.csv"))):
    name = os.path.basename(f)[:-4]; K = float(name.split("_K")[1].split("_")[0]); chi = int(name.split("_chi")[1])
    if (K, chi) not in dep:
        continue
    new = {int(r["n"]): float(r["X"]) for r in csv.DictReader(l for l in open(f) if not l.startswith("#"))}
    D = dep[(K, chi)]; ns = sorted(set(new) & set(D))
    d = {n: abs(new[n] - D[n][0]) for n in ns}
    first = {t: next((n for n in ns if d[n] > t), None) for t in (1e-8, 1e-4, 1e-2)}
    row = dict(K=K, chi=chi, run=D[ns[0]][1], first_1e8=first[1e-8], first_1e4=first[1e-4], first_1e2=first[1e-2],
               max_dX_n12=max(d[n] for n in ns if n <= 12), max_dX_n24=max(d[n] for n in ns if n <= 24))
    out.append(row); print(row)
with open(os.path.join(HERE, "machine_check.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
