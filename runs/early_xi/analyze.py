"""Prediction of the correlation length from the twisted readout at a fixed early step (Paper 1b, referee 1(b)).

Input: out/E_K*_chi*.csv from tw_run.jl (columns n, X, split_pi; split_pi = delta_pi read every four steps on the
twisted cylinder of circumference 2 a_n), and the measured xi of the same (K, chi) from the deposited runs
(sixvertex_sg_xi_predict.csv, cut 0.02, rate fixed; argv[1] or the collaboration-repository path).
g_pi = -delta_pi / (2 sqrt2 pi) [Eq. (13)], g_d(n) = (sqrt2)^(2-K) g_pi on the standard cylinder, and
xi_pred(n) = L_n/(2pi) [sqrt2 kappa_SG(K)/|g_d(n)|]^(1/(2-K)) [Eq. (12)], with no fit.
Output: early_xi.csv (one row per run and readout step n = 16, 20, 24).
"""
import csv, glob, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "..", "Otsuki", "sixvertex", "data", "sixvertex_sg_xi_predict.csv")


def kappa_sg(K):
    p = K / (2 - K)
    return (math.gamma(p / (p + 1)) / math.gamma(1 / (p + 1)) / math.pi
            * (math.sqrt(math.pi) * math.gamma((p + 1) / 2) / (2 * math.gamma(p / 2))) ** (2 / (p + 1)))


def xi_pred(K, n, g):
    L = 2 * math.sqrt(2) * math.sqrt(2) ** n
    return L / (2 * math.pi) * (math.sqrt(2) * kappa_sg(K) / abs(g)) ** (1 / (2 - K))


meas = {}
for r in csv.DictReader(open(SRC)):
    if r["cut"] == "0.02" and r["rate"] == "fixed" and float(r["mu"]) == 0 and r["xi_meas"] and \
            r["source"] in ("Yamazaki chi scan", "Yamazaki B2"):
        meas[(float(r["K"]), int(r["chi"]))] = float(r["xi_meas"])
rows = []
for f in sorted(glob.glob(os.path.join(HERE, "out", "E_K*_chi*.csv"))):
    name = os.path.basename(f)[:-4]
    K = float(name.split("_K")[1].split("_")[0]); chi = int(name.split("_chi")[1])
    d = {int(r["n"]): r for r in csv.DictReader(l for l in open(f) if not l.startswith("#"))}
    for n in (16, 20, 24):
        if n not in d or (K, chi) not in meas:
            continue
        gpi = -float(d[n]["split_pi"]) / (2 * math.sqrt(2) * math.pi)
        g = math.sqrt(2) ** (2 - K) * gpi
        xp = xi_pred(K, n, g)
        rows.append(dict(K=K, chi=chi, n=n, X=float(d[n]["X"]), g_pi=gpi, g_d=g, xi_pred=xp,
                         xi_meas=meas[(K, chi)], ratio=meas[(K, chi)] / xp))
with open(os.path.join(HERE, "early_xi.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for K in sorted({r["K"] for r in rows}):
    for n in (16, 20, 24):
        v = np.array([r["ratio"] for r in rows if r["K"] == K and r["n"] == n])
        if len(v):
            print(f"K={K} n={n}: {len(v)} runs, measured/predicted xi median {np.median(v):.3f}, range {v.min():.3f}-{v.max():.3f}")
    xs = [r for r in rows if r["K"] == K and r["n"] == 20]
    lp = np.log([r["xi_pred"] for r in xs]); lm = np.log([r["xi_meas"] for r in xs])
    if len(xs) > 2:
        print(f"   n=20: correlation of ln xi_pred and ln xi_meas over chi: {np.corrcoef(lp, lm)[0,1]:.3f}; "
              f"spread of ln xi_meas {np.std(lm):.2f}, of ln(meas/pred) {np.std(lm-lp):.2f}")
