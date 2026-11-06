"""Analysis of the single-environment baseline (batches 1 and 2).

Usage:  python3 analyze.py [DATA_DIR ...]   (default: ../data, the compressed copies of both batches)
Each DATA_DIR contains <SYM>_K<K>_chi<chi>/levels.csv and steps.csv written by run.jl.
Writes o2_fits.csv and u1_fits.csv next to this script and prints the tables.

O(2): X(n) = 1 - 4K x_{0,1}(n) is fitted from n=12 to the step before |X| first exceeds 0.1 with
      c0 + c1 t + A r^{n-n_e} (+ B (2^{2-4K})^{n-12} for K < 3/4), t = (n - (12+n_e)/2)/10,
      as in docs/03_o2/3_2_growth_rate.  The current level X_J = 1 - x_J (x_J: closest pair among the
      lowest three C-odd neutral levels) is fitted with the same columns, giving the amplitude ratio
      (A_J/A)/[alpha_J/(4K alpha_01)].  g_d^2(12) = -A r^{12-n_e}/(4K alpha_01).
U(1): O_1 = (x_{+1} - x_{-1})/2 is fitted over the last 10 points of the window that starts at n=15 and
      ends before max|4K x_{+-1} - 1| >= 0.1 or |c - 1| >= 0.2, with gamma^t [u cos(w t) + v sin(w t)],
      t = n - n_e, as in docs/04_weaker_symmetry/4_2_u1_current_growth.  With g_J of the J = i d(phi)
      convention, g_- = 2 Re g_J and Im g_J gives the neutral e = +-1 states complex dimensions
      +-4 pi i sqrt(K) Im g_J, i.e. eigenvalue phases |phi| = 2 pi K gamma^t |u sin(w t) - v cos(w t)|
      for the fixed tau = i/2 readout.  The measured phases of the e = +-1 pair are compared with this.
"""
import csv
import glob
import gzip
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
# Second-order coefficients alpha_i(K) (Table VII of the paper).  A copy is kept next to this script; the original
# is research_note/paper/docs/03_o2/3_4_amplitude_ratio/data/sixvertex_g2_alpha.csv in the project repository.
ALPHA = os.path.join(HERE, "sixvertex_g2_alpha.csv")


def _open(d, name):
    """levels.csv / steps.csv, or their gzip-compressed copies in ../data."""
    path = os.path.join(d, name)
    if os.path.exists(path):
        return open(path)
    return gzip.open(path + ".gz", "rt")


def load(d):
    lev, steps = {}, {}
    for r in csv.DictReader(_open(d, "levels.csv")):
        lev.setdefault(int(r["step"]), []).append(r)
    for r in csv.DictReader(_open(d, "steps.csv")):
        steps[int(r["step"])] = r
    return lev, steps


def level(rows, tag, rank):
    for r in rows:
        if tag in r["sector"] and int(r["rank"]) == rank:
            return float(r["x"])
    return float("nan")


def fit_growth(n, x, n_e, umklapp, K):
    t = (n - (12 + n_e) / 2) / 10
    best = None
    for lr in np.linspace(math.log(1.02), math.log(4), 4000):
        cols = [np.ones_like(t), t, np.exp(lr * (n - n_e))]
        if umklapp:
            cols.append((2.0 ** (2 - 4 * K)) ** (n - 12))
        M = np.vstack(cols).T
        c, *_ = np.linalg.lstsq(M, x, rcond=None)
        rss = ((M @ c - x) ** 2).sum()
        if best is None or rss < best[0]:
            best = (rss, lr, c, M)
    return best


def analyze_o2(d, K, chi, alpha):
    lev, _ = load(d)
    X = {n: 1 - 4 * K * level(rows, "(1, 2)", 1) for n, rows in lev.items()}
    out = [m for m in sorted(X) if m >= 12 and abs(X[m]) > 0.1]
    row = dict(K=K, chi=chi, last_step=max(X), max_abs_X=max(abs(v) for v in X.values()))
    if not out:
        return row                                     # no breakdown within the run (K > 2 control)
    n_e = out[0] - 1
    n = np.arange(12, n_e + 1, dtype=float)
    x = np.array([X[int(m)] for m in n])
    _, lr, c, M = fit_growth(n, x, n_e, K < 0.75, K)
    xj = []
    for m in n:
        v = sorted(float(r["x"]) for r in lev[int(m)] if "(0, 1)" in r["sector"] and int(r["rank"]) <= 3)
        pairs = [(abs(v[i] - v[j]), (v[i] + v[j]) / 2) for i in range(3) for j in range(i + 1, 3)]
        xj.append(1 - min(pairs)[1])
    cj, *_ = np.linalg.lstsq(M, np.array(xj), rcond=None)
    a01, aJ = float(alpha["m1"]), float(alpha["J"])
    row.update(n_e=n_e, r=math.exp(lr), r_over_pred=math.exp(lr) / 2 ** (2 - K), A=c[2],
               ratio_over_pred=(cj[2] / c[2]) / (aJ / (4 * K * a01)),
               gd2_n12=-c[2] * math.exp(lr * (12 - n_e)) / (4 * K * a01))
    return row


def analyze_u1(d, K, chi):
    lev, steps = load(d)
    good = []
    for m in sorted(lev):
        if m < 15:
            continue
        xp, xm = level(lev[m], "U₁](1)", 1), level(lev[m], "U₁](-1)", 1)
        c = float(steps[m]["central_charge"])
        if not (np.isfinite(xp) and np.isfinite(xm)) or max(abs(4 * K * xp - 1), abs(4 * K * xm - 1)) >= 0.1 \
                or abs(c - 1) >= 0.2:
            break
        good.append((m, (xp - xm) / 2))
    row = dict(K=K, chi=chi, window_points=len(good))
    if len(good) < 10:
        return row
    win = good[-10:]
    n_e = win[-1][0]
    t = np.array([m - n_e for m, _ in win], float)
    o = np.array([v for _, v in win])

    def resid(p):
        lg, w = p
        M = np.vstack([np.exp(lg * t) * np.cos(w * t), np.exp(lg * t) * np.sin(w * t)]).T
        uv, *_ = np.linalg.lstsq(M, o, rcond=None)
        return ((M @ uv - o) ** 2).sum(), uv

    starts = sorted(((resid((lg, w))[0], lg, w) for lg in np.linspace(-1.5, 1.5, 9)
                     for w in np.linspace(0.01, math.pi - 0.01, 35)))[:5]
    best = min((minimize(lambda p: resid(p)[0], (lg, w), method="Nelder-Mead",
                         options=dict(xatol=1e-10, fatol=1e-20, maxiter=4000)) for _, lg, w in starts),
               key=lambda res: res.fun)
    lg, w = best.x
    u, v = resid((lg, w))[1]
    gamma = math.exp(lg)
    # eigenvalue phases of the neutral e = +-1 pair (dimension closest to K) versus the prediction
    meas, pred = [], []
    for (m, _), tt in zip(win, t):
        # the e = +-1 pair is the complex-conjugate pair among the neutral levels near x = K; levels with no
        # phase (e.g. the degenerate current states at K = 1) are excluded by taking the two largest |phase|
        cand = sorted((abs(float(r["phase"])) for r in lev[m] if "U₁](0)" in r["sector"] and abs(float(r["x"]) - K) < 0.1),
                      reverse=True)[:2]
        if not cand:
            continue
        meas.append(np.mean(cand))
        pred.append(2 * math.pi * K * gamma ** tt * abs(u * math.sin(w * tt) - v * math.cos(w * tt)))
    meas, pred = np.array(meas), np.array(pred)
    phase_slope = float((meas @ pred) / (pred @ pred)) if len(pred) and pred @ pred > 0 else float("nan")
    row.update(n_e=n_e, gamma=gamma, gamma_over_sqrt2=gamma / math.sqrt(2), omega_deg=math.degrees(w),
               abs_gJ_ne=math.sqrt(K) * math.hypot(u, v) / (2 * math.pi), phase_points=len(pred),
               phase_meas_over_pred=phase_slope,
               phase_corr=float(np.corrcoef(meas, pred)[0, 1]) if len(pred) > 2 else float("nan"))
    return row


def main(dirs):
    alpha = {round(float(r["K"]), 4): r for r in csv.DictReader(open(ALPHA))}
    o2, u1 = [], []
    for base in dirs:
        for d in sorted(glob.glob(os.path.join(base, "*_K*_chi*"))):
            name = os.path.basename(d)
            sym = name.split("_K")[0]
            K = float(name.split("_K")[1].split("_")[0])
            chi = int(name.split("chi")[1])
            if sym == "CU1":
                o2.append(analyze_o2(d, K, chi, alpha.get(round(K, 4), {"m1": "nan", "J": "nan"})))
            else:
                u1.append(analyze_u1(d, K, chi))
    for rows, fname in ((o2, "o2_fits.csv"), (u1, "u1_fits.csv")):
        rows.sort(key=lambda r: (r["K"], r["chi"]))
        keys = list(dict.fromkeys(k for r in sorted(rows, key=len, reverse=True) for k in r))
        with open(os.path.join(HERE, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"== {fname}")
        for r in rows:
            print("  " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main(sys.argv[1:] or [os.path.join(HERE, "..", "data")])
