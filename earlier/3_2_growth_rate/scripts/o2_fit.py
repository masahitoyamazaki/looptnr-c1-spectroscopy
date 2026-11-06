"""Fits of the O(2) charged-level shift x(n) = 1 - 4K Delta_01(n), shared by the chapter 3 figures.

Port of Otsuki/sixvertex/scripts/sixvertex_rate_driftfit.jl (main and early fits) and
sixvertex_g2_jratio.jl (amplitude ratio of J and Delta_01). Every rate is fixed to its CFT value
except the growth rate r, so each fit is linear in the coefficients for a given r and r is found by
a one-dimensional minimisation of the residual sum of squares.

  main fit   x(n) = c + d t + A r^(n - n_end) [+ B s^(n - n_start)],  t = (n - (n_start + n_end)/2) / 10
             s = 2^(2-4K) (second-order umklapp), kept for K < 3/4 only
  early fit  main terms + B s^n + C (-1/2)^n + D (1/2)^n from step 6 (D merged into B at K = 3/4)
"""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

MAIN_WINDOW = (12, 0.10)
SPREAD_WINDOWS = [(n0, xmax) for n0 in (10, 12, 14) for xmax in (0.05, 0.10)]
EARLY_WINDOW = (6, 0.10)
UMKLAPP_K_MAX = 0.75          # B s^n enters x where 2^(2-4K) > 1/2
RATE_RANGE = (math.log(1.02), math.log(4.0))


def pred_rate(K):
    return 2.0 ** (2 - K)


def umklapp_rate(K):
    return 2.0 ** (2 - 4 * K)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def verify_inputs(root):
    """Check the SHA-256 of the inputs listed in provenance.json ("o2_refit" block if present).

    Returns False when data/cft/o2_levels.csv is absent (a replacement CFT directory without the
    exported levels), so the caller can skip this analysis."""
    root = Path(root)
    if not (root / "data/cft/o2_levels.csv").is_file():
        return False
    meta = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    for row in meta.get("o2_refit", meta)["inputs"]:
        path = root / row["path"]
        assert path.is_file(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path
    return True


def load_runs(cft_dir):
    """Runs and their levels from data/cft/{o2_runs.csv, o2_levels.csv}, keyed by run id."""
    cft_dir = Path(cft_dir)
    runs = {}
    with open(cft_dir / "o2_runs.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            runs[row["run"]] = dict(row, chi=int(row["chi"]), K=float(row["K"]), steps={})
    with open(cft_dir / "o2_levels.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            st = runs[row["run"]]["steps"].setdefault(int(row["step"]), {})
            st.setdefault(row["sector"], []).append((int(row["level"]), float(row["x"])))
    for r in runs.values():
        for st in r["steps"].values():
            for sec in st:
                st[sec] = [x for _, x in sorted(st[sec])]
    return runs


def jpair(levels01):
    """J, Jbar: the closer pair among the three lowest (0,1) levels; NaN when fewer than three."""
    if len(levels01) < 3:
        return math.nan
    t = sorted(levels01[:3])
    return (t[0] + t[1]) / 2 if t[1] - t[0] < t[2] - t[1] else (t[1] + t[2]) / 2


def series(run):
    """Steps n, x = 1 - 4K Delta_01 and x_J = 1 - Delta_J of one run (steps with a (1,2) level)."""
    K = run["K"]
    ns, x, xJ = [], [], []
    for n in sorted(run["steps"]):
        st = run["steps"][n]
        if not st.get("(1,2)"):
            continue
        ns.append(n)
        x.append(1 - 4 * K * st["(1,2)"][0])
        xJ.append(1 - jpair(st.get("(0,1)", [])))
    return np.array(ns), np.array(x), np.array(xJ)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
# A term is ("const",), ("lin",) or ("exp", base(r), ref) with ref "start" or "end".
def main_terms(K):
    t = [("const",), ("lin",), ("exp", lambda r: r, "end")]
    # the stored K carries ~1e-13 of rounding from c = 2 sin(pi/(4K)); K = 3/4 itself has no B
    if K < UMKLAPP_K_MAX - 1e-6:
        t.append(("exp", lambda r, s=umklapp_rate(K): s, "start"))
    return t


def early_terms(K):
    t = [("const",), ("lin",), ("exp", lambda r: r, "end"),
         ("exp", lambda r, s=umklapp_rate(K): s, "start"), ("exp", lambda r: -0.5, "start")]
    if abs(umklapp_rate(K) - 0.5) > 0.01:
        t.append(("exp", lambda r: 0.5, "start"))
    return t


def control_terms(K):
    """K >= 2: c + d n + C (-1/2)^n + D (1/2)^n + A (2^(2-K))^n, all rates fixed."""
    return [("const",), ("lin",), ("exp", lambda r: -0.5, "start"), ("exp", lambda r: 0.5, "start"),
            ("exp", lambda r, a=pred_rate(K): a, "start")]


def design(ns, terms, r):
    t = (ns - (ns[0] + ns[-1]) / 2) / 10
    cols = []
    for tm in terms:
        if tm[0] == "const":
            cols.append(np.ones(len(ns)))
        elif tm[0] == "lin":
            cols.append(t)
        else:
            nref = ns[0] if tm[2] == "start" else ns[-1]
            cols.append(tm[1](r) ** (ns - nref).astype(float))
    return np.column_stack(cols)


def term_value(tm, beta, r, wns, n):
    """Value of one fitted term at step n (n may be fractional for exponential envelopes)."""
    if tm[0] == "const":
        return beta
    if tm[0] == "lin":
        return beta * (n - (wns[0] + wns[-1]) / 2) / 10
    nref = wns[0] if tm[2] == "start" else wns[-1]
    return beta * tm[1](r) ** (n - nref)


def term_envelope(tm, beta, r, wns, n):
    nref = wns[0] if tm[2] == "start" else wns[-1]
    return abs(beta) * abs(tm[1](r)) ** (n - nref)


def solve(ns, x, terms, r):
    X = design(ns, terms, r)
    beta, *_ = np.linalg.lstsq(X, x, rcond=None)
    res = x - X @ beta
    return float(res @ res), beta, res


def window(ns, x, n0, xmax):
    """Steps n >= n0 up to the last one before |x| first exceeds xmax."""
    idx = []
    for i, n in enumerate(ns):
        if n < n0:
            continue
        if abs(x[i]) > xmax:
            break
        idx.append(i)
    return np.array(idx, dtype=int)


def best_log_rate(ns, x, terms):
    f = lambda th: solve(ns, x, terms, math.exp(th))[0]
    grid = np.linspace(*RATE_RANGE, 400)
    vals = [f(th) for th in grid]
    i = int(np.argmin(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    opt = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
    return opt.x


def profile_interval(ns, x, terms, th, rss, p, maxstep=3.0):
    """Interval of log r with RSS <= RSS_min (1 + 1/(N - p)); NaN where it does not close."""
    thr = rss * (1 + 1 / (len(ns) - p))
    f = lambda v: solve(ns, x, terms, math.exp(v))[0]

    def edge(sign):
        lo, st = th, 0.005
        while True:
            hi = lo + sign * st
            if abs(hi - th) > maxstep:
                return math.nan
            if f(hi) > thr:
                break
            lo, st = hi, st * 1.6
        for _ in range(30):
            m = (lo + hi) / 2
            if f(m) > thr:
                hi = m
            else:
                lo = m
        return (lo + hi) / 2

    return edge(-1), edge(+1)


def fit_rate(ns_all, x_all, n0, xmax, terms):
    """Fit with r free on one window; None when fewer than p + 3 points."""
    idx = window(ns_all, x_all, n0, xmax)
    p = len(terms) + 1
    if len(idx) - p < 3:
        return None
    ns, x = ns_all[idx], x_all[idx]
    th = best_log_rate(ns, x, terms)
    rss, beta, res = solve(ns, x, terms, math.exp(th))
    lo, hi = profile_interval(ns, x, terms, th, rss, p)
    return dict(r=math.exp(th), lo=math.exp(lo), hi=math.exp(hi), rms=math.sqrt(rss / (len(ns) - p)),
                N=len(ns), ns=ns, x=x, idx=idx, beta=beta, res=res, rss=rss, terms=terms, p=p)


def main_fit(K, ns, x):
    """Main fit and the spread of r over SPREAD_WINDOWS; lo_all, hi_all combine 1 sigma and the spread."""
    terms = main_terms(K)
    m = fit_rate(ns, x, *MAIN_WINDOW, terms)
    if m is None:
        return None
    vals = [f["r"] for f in (fit_rate(ns, x, *w, terms) for w in SPREAD_WINDOWS) if f is not None]
    m.update(lo_all=min(m["lo"], min(vals)), hi_all=max(m["hi"], max(vals)),
             spread=(min(vals), max(vals)), umklapp=len(terms) == 4)
    return m


def early_fit(K, ns, x):
    """Fit from step 6 with all decaying terms (K < 2), or the fixed-rate control fit (K >= 2)."""
    if K < 2:
        f = fit_rate(ns, x, *EARLY_WINDOW, early_terms(K))
        names = ["c", "d n", "A", "B", "C"] + (["D"] if len(f["terms"]) == 6 else [])
        f.update(names=names, rms=math.sqrt(f["rss"] / (f["N"] - len(f["terms"]) - 1)))
        return f
    terms = control_terms(K)
    idx = window(ns, x, EARLY_WINDOW[0], math.inf)
    rss, beta, res = solve(ns[idx], x[idx], terms, 1.0)
    return dict(r=math.nan, ns=ns[idx], x=x[idx], idx=idx, beta=beta, res=res, rss=rss, terms=terms,
                N=len(idx), rms=math.sqrt(rss / (len(idx) - len(terms))), names=["c", "d n", "C", "D", "A"])


def umklapp_at_main_start(K, ns, x):
    """|B s^(12 - 6)| of the early fit: size of the umklapp term of x at the main window start."""
    f = fit_rate(ns, x, *EARLY_WINDOW, early_terms(K))
    return abs(f["beta"][3] * umklapp_rate(K) ** (MAIN_WINDOW[0] - f["ns"][0])), f["rms"]


def amplitude_ratio(K, ns, x, xJ, n0, xmax):
    """Main fit of x on one window, then x_J on the same steps with the same terms and r: A_J / A."""
    terms = main_terms(K)
    m = fit_rate(ns, x, n0, xmax, terms)
    if m is None:
        return None
    y = xJ[m["idx"]]
    if not np.all(np.isfinite(y)):
        return None
    rssJ, betaJ, _ = solve(m["ns"], y, terms, m["r"])
    return dict(m=m, y=y, betaJ=betaJ, ratio=betaJ[2] / m["beta"][2], A=m["beta"][2], AJ=betaJ[2],
                rmsJ=math.sqrt(rssJ / (len(y) - len(terms))))


def load_alpha(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def predicted_ratio(K, alpha):
    a = min(alpha, key=lambda row: abs(row["K"] - K))
    if abs(a["K"] - K) > 1e-6:
        raise ValueError(f"no alpha row at K = {K}")
    return a["J"] / (4 * K * a["m1"])
