# Refit of the O(2) charged-level deviation X(n) = 1 - 4K Delta_01(n) of paper 1a (Eq. (14), Fig. 2, Table IV)
# with the first-order term of the spin-+-2, dimension-4 operator J^3 Jbar + J Jbar^3 that Eq. (14) lacks, its
# effect on the n >= 12 growth fit and amplitude ratio, and a test of whether a component of that form in the
# data has the level pattern of J^3 Jbar.
#
# J^3 Jbar: |g| scales as 2^{-n} and the phase of a spin-s coupling rotates by s * 135 deg per step, i.e. by
# 270 deg for spin 2, so its first-order term in any level is
#     2^{-m} [E_c cos(3 pi m / 2) + E_s sin(3 pi m / 2)],   m = n - n_start.
# With g_{J Jbar^3} = g_{J^3 Jbar}^* (real tensors) the real part of the shift of level n is S_n G,
# G = g_{J^3 Jbar} + g_{J Jbar^3}, S_n = (M_{J^3 Jbar} + M_{J Jbar^3}) / 2 (sixvertex_perturbation_results, tab:MX).
# A readout effect common to all levels (eta-like) would instead give a component proportional to x_n.
#
# Data (read only): the single-environment runs of the paper,
#   Yamazaki/runs/cpu_baseline_20261106/data/CU1_K<K>_chi<chi>/levels.csv.gz  (K = 0.75 ... 2.5)
#   Yamazaki/runs/followup_B_20261106/data/B8_K0.6_chi<chi>/levels.csv.gz     (K = 0.6)
# Fit conventions follow Yamazaki/runs/cpu_baseline_20261106/analysis/paper_figures.py (fig_terms, o2_fit).
#
# Usage (from the repository root):
#   uv run --no-project --with numpy --with scipy --with matplotlib python Otsuki/sixvertex/scripts/sixvertex_o2_j3jbar_refit.py
# Outputs: Otsuki/sixvertex/data/sixvertex_o2_j3jbar_{early,terms,main,levels,hyp}.csv,
#          Otsuki/sixvertex/figure/sixvertex_o2_j3jbar_{terms,residuals,levels,mainfit}.png,
#          Otsuki/sixvertex/logs/sixvertex_o2_j3jbar_refit_report.log

import csv
import gzip
import math
import re
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2 as chi2dist
from scipy.stats import f as fdist

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MAIN = REPO / "runs/main"
FOLLOW = REPO / "runs/additional/data"
ALPHA = REPO / "runs/main/analysis/sixvertex_g2_alpha.csv"
OUT_DATA = HERE
OUT_FIG = HERE
OUT_LOG = HERE

KS = (0.6, 0.75, 1.0, 1.25, 1.5, 1.75, 2.5)
CHIS = (24, 32, 40, 48, 64)
OMEGA = 1.5 * math.pi
RATE_RANGE = (math.log(1.02), math.log(4.0))
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def run_dir(K, chi):
    if abs(K - 0.6) < 1e-9:
        return FOLLOW / f"B8_K0.6_chi{chi}"
    return MAIN / "data" / f"CU1_K{K:.4f}_chi{chi}"


def load(K, chi):
    """step -> sector '(a,b)' -> list of x sorted by rank."""
    d = run_dir(K, chi)
    f = d / "levels.csv.gz"
    if not f.is_file():
        return None
    out = {}
    with gzip.open(f, "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sec = "(" + ",".join(re.findall(r"\d+", r["sector"].split("]")[-1])) + ")"
            out.setdefault(int(r["step"]), {}).setdefault(sec, []).append((int(r["rank"]), float(r["x"])))
    for st in out.values():
        for s in st:
            st[s] = [x for _, x in sorted(st[s])]
    return out


def jpair_rest(v01):
    """Paper rule: J is the closer pair among the three lowest (0,1) levels; the third one is returned too."""
    t = sorted(v01[:3])
    if t[1] - t[0] < t[2] - t[1]:
        return (t[0] + t[1]) / 2, t[2]
    return (t[1] + t[2]) / 2, t[0]


def closest(vals, target, k=1):
    v = sorted(vals, key=lambda y: abs(y - target))[:k]
    return sum(v) / k


def levels_at(st, K):
    """CFT values and measured x of the levels with known J^3 Jbar coefficients."""
    out = {}
    if "(1,2)" in st:
        v = st["(1,2)"]
        out["m1"] = v[0]
        if len(v) >= 5:
            out["11"] = closest(v[1:], K + 1 / (4 * K), 2)
            out["a01"] = closest(v[1:], 1 + 1 / (4 * K), 2)
    if "(2,2)" in st:
        out["m2"] = st["(2,2)"][0]
    if "(3,2)" in st:
        out["m3"] = st["(3,2)"][0]
    if "(0,1)" in st and len(st["(0,1)"]) >= 3:
        J, rest = jpair_rest(st["(0,1)"])
        out["J"] = J
        out["eO"] = rest if K < 2 else closest(st["(0,1)"][2:], K)
    if "(0,0)" in st and len(st["(0,0)"]) >= 2:
        out["eE"] = closest(st["(0,0)"][1:], K)
    return out


def cft_value(name, K):
    return {"m1": 1 / (4 * K), "m2": 1 / K, "m3": 9 / (4 * K), "eE": K, "eO": K, "J": 1.0,
            "11": K + 1 / (4 * K), "a01": 1 + 1 / (4 * K)}[name]


def S_coef(name, K):
    """Real part of the J^3 Jbar shift per unit G = g_{J^3Jbar} + g_{JJbar^3} (pair average for spin +-1)."""
    pi = math.pi
    return {"m1": pi * (K - 1) / (8 * K ** 2), "m2": pi * (K - 4) / (2 * K ** 2), "m3": 9 * pi * (K - 9) / (8 * K ** 2),
            "eE": pi * K * (4 * K - 1) / 2, "eO": pi * K * (4 * K - 1) / 2, "J": 0.0,
            "11": pi * (4 * K ** 2 - 1) * (4 * K ** 2 - K + 1) / (8 * K ** 2),
            "a01": -pi * (11 * K + 1) / (8 * K ** 2)}[name]


def series(K, chi):
    lev = load(K, chi)
    if lev is None:
        return None
    ns = np.array(sorted(n for n in lev if "(1,2)" in lev[n]))
    L = {n: levels_at(lev[n], K) for n in ns}
    names = ["m1", "m2", "m3", "eE", "eO", "J", "11", "a01"]
    xs = {nm: np.array([L[n].get(nm, math.nan) for n in ns]) for nm in names}
    X = 1 - 4 * K * xs["m1"]
    XJ = 1 - xs["J"]
    return ns, X, XJ, xs


# ---------------------------------------------------------------------------
# Fitting: terms ("const",), ("lin",), ("exp", base(r), ref), ("cos"|"sin", base(r), ref); ref "start"/"end"
# ---------------------------------------------------------------------------
def ex(base, ref="start"):
    return ("exp", (lambda r: r) if base is None else (lambda r, b=base: b), ref)


def rot(base):
    return [("cos", lambda r, b=base: b, "start"), ("sin", lambda r, b=base: b, "start")]


def design(ns, terms, r, tref):
    t = (ns - tref) / 10
    cols = []
    for tm in terms:
        if tm[0] == "const":
            cols.append(np.ones(len(ns)))
        elif tm[0] == "lin":
            cols.append(t)
        else:
            nref = ns[0] if tm[2] == "start" else ns[-1]
            m = (ns - nref).astype(float)
            b = tm[1](r) ** m
            cols.append(b if tm[0] == "exp" else b * (np.cos(OMEGA * m) if tm[0] == "cos" else np.sin(OMEGA * m)))
    return np.column_stack(cols)


def solve(ns, x, terms, r, tref):
    X = design(ns, terms, r, tref)
    beta, *_ = np.linalg.lstsq(X, x, rcond=None)
    res = x - X @ beta
    return float(res @ res), beta, res, X


def fit(ns, x, terms, r=None, tref=None):
    tref = (ns[0] + ns[-1]) / 2 if tref is None else tref
    free = r is None
    if free:
        f = lambda th: solve(ns, x, terms, math.exp(th), tref)[0]
        grid = np.linspace(*RATE_RANGE, 400)
        i = int(np.argmin([f(th) for th in grid]))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, 399)]
        r = math.exp(minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12}).x)
    rss, beta, res, X = solve(ns, x, terms, r, tref)
    p = len(terms) + (1 if free else 0)
    dof = len(ns) - p
    s2 = rss / dof
    D = np.linalg.norm(X, axis=0)
    Xs = X / D
    cov = s2 * np.linalg.inv(Xs.T @ Xs) / np.outer(D, D)
    return dict(r=r, beta=beta, se=np.sqrt(np.diag(cov)), rss=rss, rms=math.sqrt(s2), N=len(ns), p=p, dof=dof,
                ns=ns, x=x, res=res, terms=terms, tref=tref)


def E_of(f):
    ic = [i for i, tm in enumerate(f["terms"]) if tm[0] == "cos"][0]
    return complex(f["beta"][ic], f["beta"][ic + 1]), (f["se"][ic], f["se"][ic + 1])


def ftest(f0, f1):
    dp = f1["p"] - f0["p"]
    Fv = ((f0["rss"] - f1["rss"]) / dp) / (f1["rss"] / f1["dof"])
    return Fv, float(fdist.sf(Fv, dp, f1["dof"]))


# ---------------------------------------------------------------------------
# Models of X (paper Fig. 3: B only for K < 3/4; K > 2 all rates fixed, fit to the last step)
# ---------------------------------------------------------------------------
def paper_terms(K):
    t = [("const",), ("lin",)]
    if K < 2:
        t.append(ex(None, "end"))
    t += [ex(-0.5), ex(0.5)]
    if K < 0.75:
        t.append(ex(2 ** (2 - 4 * K)))
    if K > 2:
        t.append(ex(2 ** (2 - K)))
    return t


def x_models(K):
    m0 = paper_terms(K)
    mods = {"M0": m0, "M1": m0 + rot(0.5)}
    if K < 1 - 1e-9:
        mods["M2"] = mods["M1"] + [ex(2 ** (-K))]
    mods["M3"] = mods["M1"] + rot(1.0)
    return mods


def early_window(ns, X, K):
    if K > 2:
        return ns >= 6
    big = [n for n, v in zip(ns, X) if n >= 6 and abs(v) > 0.1]
    ne = big[0] - 1
    return (ns >= 6) & (ns <= ne)


def growth_window(ns, X, n0=12, cut=0.10):
    out = [n for n, v in zip(ns, X) if n >= n0 and abs(v) > cut]
    if not out:
        return None
    return (ns >= n0) & (ns <= out[0] - 1)


def level_terms(name, K, r):
    t = [("const",), ("lin",), ex(r if K < 2 else 2 ** (2 - K), "end" if K < 2 else "start"), ex(-0.5), ex(0.5)]
    if K < 0.75:
        t.append(ex(2 ** (2 - 4 * K)))
    if name in ("eE", "eO") and 2 ** (1 - 2 * K) > 0.2:
        t.append(ex(2 ** (1 - 2 * K)))      # first-order umklapp splits the C-even / C-odd pair
    return t + rot(0.5)


# ---------------------------------------------------------------------------
def main():
    for d in (OUT_DATA, OUT_FIG, OUT_LOG):
        d.mkdir(exist_ok=True)
    alpha = {round(float(r["K"]), 4): r for r in csv.DictReader(open(ALPHA, encoding="utf-8"))}
    early_rows, term_rows, main_rows, level_rows, hyp_rows = [], [], [], [], []
    store = {}
    log("# J^3 Jbar refit, paper 1a main runs (O(2)); line numbers of the paper refer to main_1a.tex at a60de2fd")

    # ---- early fits of X at chi = 40 (Fig. 3) and the level pattern
    for K in KS:
        s = series(K, 40)
        ns, X, XJ, xs = s
        w = early_window(ns, X, K)
        fits = {}
        for name, terms in x_models(K).items():
            fits[name] = fit(ns[w], X[w], terms, r=None if K < 2 else 1.0)
        f0, f1 = fits["M0"], fits["M1"]
        log(f"\nK = {K}: early window n = {f0['ns'][0]}..{f0['ns'][-1]} (N = {f0['N']})")
        for name, fr in fits.items():
            Fv = pv = math.nan
            if name == "M1":
                Fv, pv = ftest(f0, fr)
            elif name != "M0":
                Fv, pv = ftest(f1, fr)
            E6 = E12 = math.nan
            if name != "M0":
                E, _ = E_of(fr)
                E6, E12 = abs(E), abs(E) * 0.5 ** (12 - fr["ns"][0])
            rr = fr["r"] / 2 ** (2 - K) if K < 2 else math.nan
            log(f"  {name}: rms = {fr['rms']:.3e}, r/2^(2-K) = {rr:.4f}, |E|(6) = {E6:.2e}, |E|(12) = {E12:.2e}, "
                f"F = {Fv:.2f}, p = {pv:.2g}")
            early_rows.append(dict(K=K, model=name, n_start=int(fr["ns"][0]), n_end=int(fr["ns"][-1]), N=fr["N"],
                                   p=fr["p"], rss=fr["rss"], rms=fr["rms"], r=fr["r"], r_over_pred=rr,
                                   E_abs_n6=E6, E_abs_n12=E12, F=Fv, p_value=pv))
            for i, (b, se) in enumerate(zip(fr["beta"], fr["se"])):
                tm = fr["terms"][i]
                lab = tm[0] if tm[0] in ("const", "lin", "cos", "sin") else f"exp({tm[1](fr['r']):.4g})"
                term_rows.append(dict(K=K, model=name, term=lab, coefficient=b, std_error=se))
        for lab, base in (("C (-1/2)", -0.5), ("D (+1/2)", 0.5)):
            i0 = [i for i, tm in enumerate(f0["terms"]) if tm[0] == "exp" and abs(tm[1](1.0) - base) < 1e-12][0]
            log(f"    {lab}: M0 {f0['beta'][i0]:+.3e} -> M1 {f1['beta'][i0]:+.3e} (+- {f1['se'][i0]:.1e})")

        # level pattern of the fitted (1/2, 270 deg) component
        r1 = f1["r"]
        rows = []
        for nm in ("m1", "m2", "m3", "eE", "eO", "J", "11", "a01"):
            if K == 1.0 and nm in ("11", "a01"):
                continue                      # degenerate at K = 1, split at first order by V_d (Sec. IV D)
            y = xs[nm][w]
            ok = np.isfinite(y)
            terms = level_terms(nm, K, r1)
            if ok.sum() < len(terms) + 4:
                continue
            fl = fit(ns[w][ok], y[ok], terms, r=1.0 if K > 2 else r1, tref=f1["tref"])
            E, se = E_of(fl)
            m0 = fl["ns"][0] - f1["ns"][0]          # refer E to the start of the X window
            E = E * (2.0 ** m0) * complex(math.cos(OMEGA * m0), -math.sin(OMEGA * m0))
            se = (se[0] * 2.0 ** m0, se[1] * 2.0 ** m0)
            rows.append(dict(K=K, level=nm, x_cft=cft_value(nm, K), S=S_coef(nm, K), E_re=E.real, E_im=E.imag,
                             se_re=se[0], se_im=se[1], rms=fl["rms"], N=fl["N"]))
        level_rows += rows
        # hypotheses: E_n = S_n G (J^3 Jbar), E_n = x_n H (common scale), E_n = 0
        E = np.array([complex(r["E_re"], r["E_im"]) for r in rows])
        sig = np.array([0.5 * (r["se_re"] + r["se_im"]) for r in rows])
        hyp = {}
        for hname, vec in (("J3Jbar", np.array([r["S"] for r in rows])), ("scale", np.array([r["x_cft"] for r in rows])),
                           ("zero", None)):
            if vec is None:
                chi2v, npar, G = float(np.sum(np.abs(E) ** 2 / sig ** 2)), 0, 0j
            else:
                wgt = 1 / sig ** 2
                G = complex(np.sum(wgt * vec * E.real) / np.sum(wgt * vec ** 2), np.sum(wgt * vec * E.imag) / np.sum(wgt * vec ** 2))
                chi2v, npar = float(np.sum(np.abs(E - vec * G) ** 2 / sig ** 2)), 2
            dof = 2 * len(rows) - npar
            hyp[hname] = dict(chi2=chi2v, dof=dof, p=float(chi2dist.sf(chi2v, dof)), G=G)
            hyp_rows.append(dict(K=K, hypothesis=hname, chi2=chi2v, dof=dof, p_value=hyp[hname]["p"],
                                 amp_re=G.real, amp_im=G.imag, n_levels=len(rows)))
        log("  level pattern: " + ", ".join(f"{r['level']} |E| = {abs(complex(r['E_re'], r['E_im'])):.1e}"
                                            f" (+-{0.5 * (r['se_re'] + r['se_im']):.0e})" for r in rows))
        log("  hypotheses: " + ", ".join(f"{h}: chi2/dof = {v['chi2']:.1f}/{v['dof']} (p = {v['p']:.2g})" for h, v in hyp.items()))
        store[K] = dict(ns=ns, X=X, w=w, fits=fits, rows=rows, hyp=hyp)

    # ---- growth fits (n >= 12) for all chi: paper model vs + J^3 Jbar (+ V_d descendant for K < 1)
    log("\n# growth fit n >= 12 (paper Table IV conventions), r/2^(2-K) and normalized amplitude ratio")
    for K in KS:
        if K > 2:
            continue
        for chi in CHIS:
            s = series(K, chi)
            if s is None:
                continue
            ns, X, XJ, _ = s
            if growth_window(ns, X) is None:
                continue
            pred = float(alpha[round(K, 4)]["J"]) / (4 * K * float(alpha[round(K, 4)]["m1"]))
            res = {}
            for name, extra in (("M0", []), ("M1", rot(0.5)), ("M2", rot(0.5) + ([ex(2 ** (-K))] if K < 1 else []))):
                terms = [("const",), ("lin",), ex(None, "end")] + ([ex(2 ** (2 - 4 * K))] if K < 0.75 else []) + extra
                vals = []
                for n0 in (10, 12, 14):
                    for cut in (0.05, 0.10):
                        gw = growth_window(ns, X, n0, cut)
                        if gw is None or gw.sum() < len(terms) + 4:
                            continue
                        fw = fit(ns[gw], X[gw], terms)
                        _, bJ, _, _ = solve(ns[gw], XJ[gw], terms, fw["r"], fw["tref"])
                        vals.append((n0, cut, fw["r"], (bJ[2] / fw["beta"][2]) / pred, int(ns[gw][-1])))
                main = [v for v in vals if v[0] == 12 and v[1] == 0.10]
                if not main:
                    continue
                m = main[0]
                res[name] = m
                main_rows.append(dict(K=K, chi=chi, model=name, n_end=m[4], r=m[2], r_over_pred=m[2] / 2 ** (2 - K),
                                      r_lo=min(v[2] for v in vals) / 2 ** (2 - K), r_hi=max(v[2] for v in vals) / 2 ** (2 - K),
                                      ratio=m[3], ratio_lo=min(v[3] for v in vals), ratio_hi=max(v[3] for v in vals)))
            if res:
                log(f"  K = {K}, chi = {chi}: " + ", ".join(
                    f"{k}: r = {v[2]:.4f} (r/2^(2-K) = {v[2] / 2 ** (2 - K):.4f}), ratio = {v[3]:.4f}" for k, v in res.items()))

    def wcsv(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)

    wcsv(OUT_DATA / "sixvertex_o2_j3jbar_early.csv", early_rows)
    wcsv(OUT_DATA / "sixvertex_o2_j3jbar_terms.csv", term_rows)
    wcsv(OUT_DATA / "sixvertex_o2_j3jbar_main.csv", main_rows)
    wcsv(OUT_DATA / "sixvertex_o2_j3jbar_levels.csv", level_rows)
    wcsv(OUT_DATA / "sixvertex_o2_j3jbar_hyp.csv", hyp_rows)
    (OUT_LOG / "sixvertex_o2_j3jbar_refit_report.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    plots(store, main_rows)


# ---------------------------------------------------------------------------
# Figures (reference categorical palette, fixed order)
# ---------------------------------------------------------------------------
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300", "#e34948"]
INK, MUTED = "#1f1f1f", "#8a8a8a"
LEVEL_LABEL = {"m1": "$m=1$", "m2": "$m=2$", "m3": "$m=3$", "eE": "$e$, C even", "eO": "$e$, C odd", "J": "$J$",
               "11": "$(1,1)$", "a01": "$a_{-1}|0,1\\rangle$"}


def plots(store, main_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.linewidth": 0.6})

    # (1) Fig. 3 with the J^3 Jbar term (M1)
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.0))
    lab = {"A": ("growth $A r_{\\rm d}^{n}$", S[0]), "B": ("umklapp $(2^{2-4K})^n$", S[1]),
           "C": ("dim 4, spin $\\pm4$: $(-1/2)^n$", S[2]), "D": ("dim 4, spin 0: $(1/2)^n$", S[3]),
           "E": ("$J^3\\bar J$: $2^{-n}$, $270^\\circ$/step", S[4])}
    for ax, K in zip(axes.flat, KS):
        d = store[K]
        f1 = d["fits"]["M1"]
        ns, X = d["ns"], d["X"]
        wns = f1["ns"]
        bg = f1["beta"][0] + f1["beta"][1] * (ns - f1["tref"]) / 10
        stop = next((n for n, v in zip(ns, X) if n >= 1 and abs(v) > 0.3), ns[-1] + 1)
        sel = (ns >= 1) & (ns < stop)
        ax.semilogy(ns[sel], np.abs(X - bg)[sel], "o", ms=2.8, color=INK, mec="none", alpha=0.75)
        ax.axvspan(wns[0], wns[-1], color="#000000", alpha=0.05, lw=0)
        nn = np.linspace(1, stop, 500)
        for i, tm in enumerate(f1["terms"]):
            if tm[0] in ("const", "lin", "sin"):
                continue
            if tm[0] == "cos":
                env = abs(complex(f1["beta"][i], f1["beta"][i + 1])) * 0.5 ** (nn - wns[0])
                key = "E"
            else:
                b = tm[1](f1["r"])
                key = "A" if (K < 2 and tm[2] == "end") or (K > 2 and abs(b - 2 ** (2 - K)) < 1e-12) else \
                    "C" if abs(b + 0.5) < 1e-12 else "D" if abs(b - 0.5) < 1e-12 else "B"
                nref = wns[0] if tm[2] == "start" else wns[-1]
                env = abs(f1["beta"][i]) * abs(b) ** (nn - nref)
            inside = (nn >= wns[0]) & (nn <= wns[-1])
            ax.semilogy(nn[inside], env[inside], color=lab[key][1], lw=1.5)
            ax.semilogy(nn, env, color=lab[key][1], lw=0.8, ls=":")
        ax.axhline(1e-4, color=MUTED, lw=0.8, ls="--")
        ax.set_ylim(1e-6, 1)
        ax.set_xlim(0, stop)
        ax.set_title(f"$K={K:g}$", fontsize=8.5)
        ax.grid(alpha=0.15, lw=0.4)
        ax.set_xlabel("RG step $n$")
    for ax in axes[:, 0]:
        ax.set_ylabel("$|X-c-d\\,t|$")
    la = axes.flat[-1]
    la.axis("off")
    for key in "ABCDE":
        la.plot([], [], color=lab[key][1], lw=1.5, label=lab[key][0])
    la.plot([], [], "o", color=INK, ms=3, label="data ($\\chi=40$)")
    la.plot([], [], color=MUTED, ls="--", lw=0.8, label="$10^{-4}$")
    la.legend(loc="center", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "sixvertex_o2_j3jbar_terms.png", dpi=180)
    plt.close(fig)

    # (2) residuals of the paper model, scaled by 2^{n-6}, and the J^3 Jbar term of M1
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 4.6))
    for ax, K in zip(axes.flat, KS):
        f0, f1 = store[K]["fits"]["M0"], store[K]["fits"]["M1"]
        n0 = f0["ns"][0]
        up = f0["ns"] <= n0 + 7
        m = f0["ns"][up] - n0
        sc = 2.0 ** m
        E, _ = E_of(f1)
        mm = np.arange(0, m[-1] + 1)
        pat = E.real * np.cos(OMEGA * mm) + E.imag * np.sin(OMEGA * mm)
        ax.fill_between(f0["ns"][up], -f0["rms"] * sc, f0["rms"] * sc, color=MUTED, alpha=0.15, lw=0, label="$\\pm$rms(M0)$\\times2^{n-6}$")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.plot(f0["ns"][up], f0["res"][up] * sc, "o-", ms=3.5, lw=0.8, color=S[0], label="residual of M0 $\\times2^{n-6}$")
        ax.plot(n0 + mm, pat, "s--", ms=3.5, lw=0.8, color=S[4], label="$J^3\\bar J$ term of M1 $\\times2^{n-6}$")
        lim = 1.2 * max(np.max(np.abs(f0["res"][up] * sc)), np.max(np.abs(pat)), f0["rms"] * sc[-1])
        ax.set_ylim(-lim, lim)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.set_title(f"$K={K:g}$", fontsize=8.5)
        ax.grid(alpha=0.15, lw=0.4)
        ax.set_xlabel("RG step $n$")
    axes.flat[-1].axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(h, l, loc="center", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "sixvertex_o2_j3jbar_residuals.png", dpi=180)
    plt.close(fig)

    # (3) level pattern: E_n projected on the phase of the J^3 Jbar fit, against S_n G and x_n H
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.2))
    for ax, K in zip(axes.flat, KS):
        rows, hyp = store[K]["rows"], store[K]["hyp"]
        G = hyp["J3Jbar"]["G"]
        ph = G / abs(G) if abs(G) > 0 else 1
        xi = np.arange(len(rows))
        Ep = np.array([(complex(r["E_re"], r["E_im"]) / ph).real for r in rows])
        Eo = np.array([(complex(r["E_re"], r["E_im"]) / ph).imag for r in rows])
        se = np.array([0.5 * (r["se_re"] + r["se_im"]) for r in rows])
        H = hyp["scale"]["G"]
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.errorbar(xi - 0.12, Ep, yerr=se, fmt="o", ms=5, color=S[0], mec="white", mew=1, elinewidth=1, label="data, in phase with $G$")
        ax.errorbar(xi + 0.12, Eo, yerr=se, fmt="o", ms=4, color=MUTED, mfc="white", elinewidth=0.8, label="data, $90^\\circ$ out of phase")
        ax.plot(xi, [r["S"] * abs(G) for r in rows], "_", ms=14, mew=2, color=S[4], label="$J^3\\bar J$ fit $S_nG$")
        ax.plot(xi, [(r["x_cft"] * H / ph).real for r in rows], "_", ms=14, mew=2, color=S[1], label="common scale $x_nH$")
        ax.set_xticks(xi)
        ax.set_xticklabels([LEVEL_LABEL[r["level"]] for r in rows], rotation=60, fontsize=7)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        hj, hs = hyp["J3Jbar"], hyp["scale"]
        ax.set_title(f"$K={K:g}$: $\\chi^2$ {hj['chi2']:.0f} vs {hs['chi2']:.0f} (dof {hj['dof']})", fontsize=7.5)
        ax.grid(alpha=0.15, lw=0.4, axis="y")
    axes[0, 0].set_ylabel("$E_n$ at $n=6$ (projected)")
    axes[1, 0].set_ylabel("$E_n$ at $n=6$ (projected)")
    axes.flat[-1].axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(h, l, loc="center", frameon=False, fontsize=8,
                         title="$\\chi^2$: $J^3\\bar J$ vs common scale", title_fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "sixvertex_o2_j3jbar_levels.png", dpi=180)
    plt.close(fig)

    # (4) growth fit: change of r/2^{2-K} and of the amplitude ratio when J^3 Jbar (and the descendant) is added
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))
    chis = sorted({r["chi"] for r in main_rows})
    for j, chi in enumerate(chis):
        for mname, mk in (("M1", "o"), ("M2", "s")):
            base = {r["K"]: r for r in main_rows if r["chi"] == chi and r["model"] == "M0"}
            rr = [r for r in main_rows if r["chi"] == chi and r["model"] == mname and r["K"] in base]
            if mname == "M2":
                rr = [r for r in rr if r["K"] < 1]
            if not rr:
                continue
            Ks = np.array([r["K"] for r in rr]) + 0.012 * (j - 2)
            axes[0].plot(Ks, [100 * (r["r_over_pred"] - base[r["K"]]["r_over_pred"]) for r in rr], mk, ms=4.5,
                         color=S[j], mfc=S[j] if mname == "M1" else "white", label=f"$\\chi={chi}$" if mname == "M1" else None)
            axes[1].plot(Ks, [100 * (r["ratio"] - base[r["K"]]["ratio"]) for r in rr], mk, ms=4.5,
                         color=S[j], mfc=S[j] if mname == "M1" else "white")
    for ax in axes:
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xlabel("$K$")
        ax.grid(alpha=0.15, lw=0.4)
    axes[0].set_ylabel("change of $r_{\\rm d}/2^{2-K}$ [%]")
    axes[1].set_ylabel("change of normalized $A_J/A$ [%]")
    axes[0].plot([], [], "o", color=MUTED, label="+ $J^3\\bar J$ (M1)")
    axes[0].plot([], [], "s", color=MUTED, mfc="white", label="+ $V_{\\rm d}$ descendant (M2, $K<1$)")
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "sixvertex_o2_j3jbar_mainfit.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
