"""Effective Luttinger parameter K_n and common scale eta from the saved O(2) spectra.

Usage:  python3 drift.py DATA_DIR [DATA_DIR ...]      (writes drift_summary.csv, drift_steps.csv)

At each step two levels fix (K_n, eta) with x_i = (1 + eta) x_i^CFT(K_n):
  x_q = x_{0,1}                       CFT 1/(4K)   (lowest level of charge 1)
  x_e = (x_E + x_O)/2                 CFT K        (C-even and C-odd combinations of e = +-1, averaged,
                                                   which cancels the first-order umklapp splitting)
  K_n = sqrt(x_e / x_q) / 2,   1 + eta = 2 sqrt(x_e x_q).
Levels not used for the estimate are then predicted:
  x_{0,2} = (1+eta)/K_n,  x_{0,3} = 9(1+eta)/(4K_n),  current x_J = 1+eta   (C-odd neutral, K independent).
The drift is measured in the 'floor' window, after the initial transient (n >= 12, where the dimension-4 terms are below the floor) and before the
fitted dimerization term of analyze.py (o2_fits.csv) exceeds 1e-4; a linear fit of K_n(n) there gives
the direct marginal source dK/dn.  For runs without growth (K > 2) the whole run after n >= 12 is used.
"""
import csv
import glob
import math
import os
import sys

import numpy as np

from analyze import load

HERE = os.path.dirname(os.path.abspath(__file__))
GROWTH = {(float(r["K"]), int(r["chi"])): (float(r["A"]), float(r["r"]), int(r["n_e"]))
          for r in csv.DictReader(open(os.path.join(HERE, "o2_fits.csv"))) if r.get("A")}


def levels(rows, tag):
    return sorted(float(r["x"]) for r in rows if tag in r["sector"])


def closest(vals, target):
    return min(vals, key=lambda v: abs(v - target)) if vals else float("nan")


def analyze(d, K, chi):
    lev, _ = load(d)
    steps = []
    for n in sorted(lev):
        rows = lev[n]
        q1, q2, q3 = (levels(rows, f"({m}, 2)") for m in (1, 2, 3))
        even, odd = levels(rows, "(0, 0)")[1:], levels(rows, "(0, 1)")
        if not (q1 and q2 and q3 and even and odd):
            continue
        xq = q1[0]
        xE = closest(even, K)
        # C-odd neutral: the current pair sits at 1, the e = +-1 combination at K
        xO = closest([v for v in odd if abs(v - 1) > 0.02] or odd, K) if abs(K - 1) > 0.05 else closest(odd, K)
        xe = (xE + xO) / 2
        if not (xq > 0 and xe > 0):
            continue
        Kn = math.sqrt(xe / xq) / 2
        s = 2 * math.sqrt(xe * xq)                        # 1 + eta
        xJ = closest(odd, 1.0)
        steps.append(dict(K=K, chi=chi, n=n, X=1 - 4 * K * xq, K_n=Kn, eta=s - 1,
                          res_q2=q2[0] - s / Kn, res_q3=q3[0] - 9 * s / (4 * Kn), res_J=xJ - s))
    # floor window: n >= 8 and before the fitted growth term A r^(n - n_e) (o2_fits.csv) exceeds 1e-4
    fit = GROWTH.get((K, chi))
    if fit is None:
        floor = [r for r in steps if r["n"] >= 12]
    else:
        A, rr, ne = fit
        floor = [r for r in steps if r["n"] >= 12 and abs(A) * rr ** (r["n"] - ne) < 1e-4]
    summ = dict(K=K, chi=chi, floor_points=len(floor))
    if len(floor) >= 4:
        n = np.array([r["n"] for r in floor], float)
        kn = np.array([r["K_n"] for r in floor])
        slope, icpt = np.polyfit(n, kn, 1)
        summ.update(n_first=int(n[0]), n_last=int(n[-1]), K_n_mean=kn.mean(), dK_per_step=slope,
                    K_n_scatter=float(np.std(kn - (slope * n + icpt))),
                    eta_mean=float(np.mean([r["eta"] for r in floor])),
                    rms_res_q2=float(np.sqrt(np.mean([r["res_q2"] ** 2 for r in floor]))),
                    rms_res_q3=float(np.sqrt(np.mean([r["res_q3"] ** 2 for r in floor]))),
                    rms_res_J=float(np.sqrt(np.mean([r["res_J"] ** 2 for r in floor]))))
    return summ, steps


def main(dirs):
    summary, allsteps = [], []
    for base in dirs:
        for d in sorted(glob.glob(os.path.join(base, "CU1_K*_chi*"))):
            name = os.path.basename(d)
            K = float(name.split("_K")[1].split("_")[0])
            chi = int(name.split("chi")[1])
            s, st = analyze(d, K, chi)
            summary.append(s)
            allsteps += st
    summary.sort(key=lambda r: (r["K"], r["chi"]))
    for rows, fname in ((summary, "drift_summary.csv"), (allsteps, "drift_steps.csv")):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(HERE, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    for r in summary:
        print("  " + "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main(sys.argv[1:])
