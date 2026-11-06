"""Compact-boson spectrum by symmetry sector and level comparison (used by b9_validation.py).
Compact boson prediction: x = e^2/(4K) + K m^2 + N + Nbar, C: (e,m)->(-e,-m), oscillators odd.
"""
import csv, glob, os, re, sys, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NMAX, MMAX, EMAX = 4, 3, 4
XCUT = 4.0

# chiral Fock states: list of (N, nparts) with multiplicity
def partitions(n, maxpart=None):
    if maxpart is None: maxpart = n
    if n == 0: yield []; return
    for k in range(min(n, maxpart), 0, -1):
        for rest in partitions(n - k, k): yield [k] + rest
FOCK = [(N, len(p)) for N in range(NMAX + 1) for p in partitions(N)]

def tower(x0, parity_split=False):
    out = []
    for N, a in FOCK:
        for Nb, b in FOCK:
            x = x0 + N + Nb
            if x <= XCUT: out.append((x, (a + b) % 2))
    return out

def predicted(K):
    """returns dict sector-> sorted list of x."""
    e0_even, e0_odd = [], []
    for x, par in tower(0.0):
        (e0_even if par == 0 else e0_odd).append(x)
    for m in range(1, MMAX + 1):
        for x, _ in tower(K * m * m):
            e0_even.append(x); e0_odd.append(x)
    charged = {}
    for e in range(1, EMAX + 1):
        lst = []
        for m in range(-MMAX, MMAX + 1):
            lst += [x for x, _ in tower(e * e / (4 * K) + K * m * m)]
        charged[e] = sorted(lst)
    allpos = sorted(sum(charged.values(), []))
    return {
        "CU1(0,0)": sorted(e0_even), "CU1(0,1)": sorted(e0_odd),
        **{f"CU1({e},2)": charged[e] for e in (1, 2, 3)},
        "Z2(0)": sorted(e0_even + allpos), "Z2(1)": sorted(e0_odd + allpos),
        "U1(0)": sorted(e0_even + e0_odd),
        **{f"U1({q})": charged[abs(q)] for q in (1, -1, 2, -2, 3, -3)},
    }

def sector_key(s):
    m = re.search(r"CU₁\]\((\d+), (\d+)\)", s)
    if m: return f"CU1({m.group(1)},{m.group(2)})"
    m = re.search(r"U₁\]\((-?\d+)\)", s)
    if m: return f"U1({m.group(1)})"
    m = re.search(r"ℤ₂\]\((\d+)\)", s)
    if m: return f"Z2({m.group(1)})"
    return s

def comparable_k(pred, nh, gap=0.05):
    k = 0
    for i in range(1, min(nh, len(pred) - 1) + 1):
        if pred[i] - pred[i - 1] > gap: k = i
    return k

def _open(d, name):
    import gzip
    p = os.path.join(d, name)
    return open(p) if os.path.exists(p) else gzip.open(p + ".gz", "rt")


def load_run(d):
    steps = {}
    with _open(d, "steps.csv") as f:
        for r in csv.DictReader(f):
            steps[int(r["step"])] = dict(c=float(r["central_charge"]), status=r["status"],
                                         sweeps=int(r["sweeps"]), cost=float(r["final_cost"]),
                                         rg=float(r["rg_seconds"]), dims=r["dims_A"])
    lv = defaultdict(lambda: defaultdict(list))
    with _open(d, "levels.csv") as f:
        for r in csv.DictReader(f):
            lv[int(r["step"])][sector_key(r["sector"])].append(float(r["x"]))
    return steps, lv

def parse_K(tag):
    return float(re.search(r"K([\d.]+)", tag).group(1))

def analyze(d):
    tag = os.path.basename(d)
    K = parse_K(tag)
    pred = predicted(K)
    steps, lv = load_run(d)
    rows = []
    for n in sorted(steps):
        errs, prim = [], {}
        for sec, xs in lv[n].items():
            xs = sorted(xs)
            p = pred[sec]
            k = min(comparable_k(p, len(xs)), len(xs))
            if k: errs += list(np.array(xs[:k]) - np.array(p[:k]))
        L = lv[n]
        def first(sec, i=0):
            v = sorted(L.get(sec, []))
            return v[i] if len(v) > i else np.nan
        if "U1(1)" in L:
            prim = dict(x_e1=first("U1(1)"), x_e2=first("U1(2)"), x_e3=first("U1(3)"))
        elif "CU1(1,2)" in L:
            prim = dict(x_e1=first("CU1(1,2)"), x_e2=first("CU1(2,2)"), x_e3=first("CU1(3,2)"))
        else:
            prim = dict(x_e1=first("Z2(1)", 0), x_e2=np.nan, x_e3=np.nan)
        rms = float(np.sqrt(np.mean(np.square(errs)))) if errs else np.nan
        mx = float(np.max(np.abs(errs))) if errs else np.nan
        rows.append(dict(step=n, c=steps[n]["c"], rms=rms, maxerr=mx, nlev=len(errs), **prim,
                         status=steps[n]["status"], sweeps=steps[n]["sweeps"], cost=steps[n]["cost"],
                         rg=steps[n]["rg"]))
    return tag, K, rows

def summarize(tag, K, rows, ctol=5e-3, etol=1e-2):
    good = [abs(r["c"] - 1) < ctol and r["rms"] < etol for r in rows]
    best, cur, s0 = (0, None, None), 0, None
    for i, g in enumerate(good):
        if g:
            if cur == 0: s0 = rows[i]["step"]
            cur += 1
            if cur > best[0]: best = (cur, s0, rows[i]["step"])
        else: cur = 0
    L, a, b = best
    win = [r for r in rows if a is not None and a <= r["step"] <= b]
    brk = next((r["step"] for r in rows if a is not None and r["step"] > b and abs(r["c"] - 1) > 0.05), None)
    ex1 = 1 / (4 * K)
    out = dict(tag=tag, K=K, plateau_len=L, plateau=(a, b), break_step=brk, last_step=rows[-1]["step"])
    if win:
        out.update(rms_med=float(np.median([r["rms"] for r in win])), rms_min=float(min(r["rms"] for r in win)),
                   c_med=float(np.median([r["c"] for r in win])),
                   xe1_med=float(np.median([r["x_e1"] for r in win])), xe1_exact=ex1,
                   xe1_relerr=float(np.median([abs(r["x_e1"] - ex1) / ex1 for r in win])),
                   rg_med=float(np.median([r["rg"] for r in win])))
    return out

