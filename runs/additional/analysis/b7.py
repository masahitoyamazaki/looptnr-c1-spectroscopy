"""Analysis of runs B7 (only charge conjugation C imposed; U(1) broken).

Usage:  python3 b7.py OUT_DIR      (B7_C_K*_chi*/ with levels.csv, steps.csv from the workstation)

With only C imposed, the charge-(+1) and charge-(-1) primaries (x = 1/(4K)) appear as a C-even level in
the Z2(0) sector and a C-odd level in the Z2(1) sector.  Their splitting
    D(n) = x_even(n) - x_odd(n)
measures the mixing of (0,+1) and (0,-1), i.e. the U(1)-breaking couplings:
  * V_{0,+-2} (x = 1/K) mixes them at first order:      D ~ g_{0,2},    factor 2^{1 - 1/(2K)} per step;
  * V_{0,+-1} (x = 1/(4K)) only at second order:         D ~ g_{0,1}^2,  factor 2^{2 - 1/(4K)} per step.
Both are spin zero, so D is real and does not oscillate.  We fit log|D| linearly in n over the steps
n >= 10 in which the relative splitting 4K|D| lies between LO and HI (contiguous from the first such step;
default 1e-5 and 1e-2), and report the fitted factor rho with its range over (LO, HI) in
{1e-5, 1e-4} x {3e-3, 1e-2, 3e-2}.  The C-averaged level E = (x_even + x_odd)/2 and the first step at
which max |4K x - 1| >= 0.1 (n_e, as for the O(2) window) are also reported.
Writes b7.csv next to this script.
"""
import csv, glob, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "main", "analysis"))
from analyze import load   # noqa: E402


def series(d, K):
    lev, steps = load(d)
    out = {}
    for n, rows in lev.items():
        even = sorted(float(r["x"]) for r in rows if "(0)" in r["sector"])
        odd = sorted(float(r["x"]) for r in rows if "(1)" in r["sector"])
        if len(even) < 2 or not odd:
            continue
        ref = 1 / (4 * K)
        xe = min(even[1:], key=lambda x: abs(x - ref))       # skip the vacuum
        xo = min(odd, key=lambda x: abs(x - ref))
        out[n] = dict(xe=xe, xo=xo, D=xe - xo, E=(xe + xo) / 2, c=float(steps[n]["central_charge"]))
    return out


def fit(s, K, lo, hi):
    ns = sorted(n for n in s if n >= 10)
    win = []
    for n in ns:
        rel = 4 * K * abs(s[n]["D"])
        if lo < rel < hi:
            win.append(n)
        elif win:
            break
    if len(win) < 4:
        return None
    n = np.array(win, float)
    y = np.log(np.abs([s[m]["D"] for m in win]))
    p = np.polyfit(n, y, 1)
    signs = {int(np.sign(s[m]["D"])) for m in win}
    return dict(rho=math.exp(p[0]), n1=win[0], n2=win[-1], sign_const=len(signs) == 1,
                rms=float(np.sqrt(np.mean((np.polyval(p, n) - y) ** 2))))


def main(out):
    rows = []
    print(f"{'run':20s} {'n_e':>4s} {'window':>7s} {'rho':>7s} {'range':>15s} {'rho/2^(1-1/2K)':>15s} {'2^(2-1/4K)':>10s} sign")
    for d in sorted(glob.glob(os.path.join(out, "B7_C_K*_chi*"))):
        name = os.path.basename(d)
        K = float(name.split("_K")[1].split("_")[0]); chi = int(name.split("chi")[1])
        s = series(d, K)
        bad = [n for n in sorted(s) if n >= 10 and (max(abs(4 * K * s[n]["xe"] - 1), abs(4 * K * s[n]["xo"] - 1)) >= 0.1
                                                  or abs(s[n]["c"] - 1) >= 0.2)]
        n_e = bad[0] - 1 if bad else max(s)
        main_fit = fit(s, K, 1e-5, 1e-2)
        var = [f for lo in (1e-5, 1e-4) for hi in (3e-3, 1e-2, 3e-2) if (f := fit(s, K, lo, hi))]
        p1 = 2 ** (1 - 1 / (2 * K)); p2 = 2 ** (2 - 1 / (4 * K))
        row = dict(K=K, chi=chi, n_e=n_e, pred_first_order_V02=p1, pred_second_order_V01=p2)
        if main_fit:
            lo_, hi_ = min(f["rho"] for f in var), max(f["rho"] for f in var)
            row.update(rho=main_fit["rho"], rho_min=lo_, rho_max=hi_, rho_over_pred=main_fit["rho"] / p1,
                       window=f"{main_fit['n1']}-{main_fit['n2']}", sign_const=main_fit["sign_const"],
                       log_rms=main_fit["rms"])
            print(f"{name:20s} {n_e:4d} {row['window']:>7s} {main_fit['rho']:7.4f} [{lo_:.4f},{hi_:.4f}] "
                  f"{main_fit['rho'] / p1:15.4f} {p2:10.3f} {'const' if main_fit['sign_const'] else 'FLIPS'}")
        else:
            print(f"{name:20s} {n_e:4d}   no window")
        rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(HERE, "b7.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    figure(out, rows)


def figure(out, rows):
    """fig_c_only.pdf: (a) |D(n)| at chi = 40 with the first-order V_{0,+-2} slope; (b) rho / 2^(1-1/(2K))."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts (journals reject Type 3)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    Ks = (0.75, 1.0, 1.25, 1.5, 1.75, 2.5)
    for i, K in enumerate(Ks):
        s = series(os.path.join(out, f"B7_C_K{K}_chi40"), K)
        ns = [n for n in sorted(s) if n >= 4]
        col = f"C{i}"
        ax[0].semilogy(ns, [4 * K * abs(s[n]["D"]) for n in ns], "o", ms=2.5, color=col, label=f"$K={K}$")
        f = fit(s, K, 1e-5, 1e-2)
        if f:
            n = np.arange(f["n1"], f["n2"] + 1)
            p1 = 2 ** (1 - 1 / (2 * K))
            ax[0].semilogy(n, 4 * K * abs(s[f["n2"]]["D"]) * p1 ** (n - f["n2"]), "-", lw=1, color=col)
    ax[0].set_xlabel("RG step $n$"); ax[0].set_ylabel(r"$4K\,|x_{\rm even}-x_{\rm odd}|$")
    ax[0].set_ylim(1e-16, 1); ax[0].legend(fontsize=6, ncol=2, loc="lower right")
    ax[0].set_title(r"(a) $C$ only, $\chi=40$", fontsize=9)
    for j, (chi, mk) in enumerate(((24, "^"), (32, "s"), (40, "o"))):
        pts = [(r["K"] + 0.03 * (j - 1), r["rho_over_pred"]) for r in rows if r["chi"] == chi and "rho" in r]
        ax[1].plot(*zip(*pts), mk, ms=4, label=rf"$\chi={chi}$")
    ax[1].axhline(1, color="k", lw=0.8, ls="--")
    ax[1].axhspan(0.97, 1.03, color="0.9", zorder=0)
    ax[1].set_xlabel("$K$"); ax[1].set_ylabel(r"$r_{0,2}/2^{1-1/(2K)}$"); ax[1].set_ylim(0.8, 1.2)
    ax[1].legend(fontsize=7); ax[1].set_title(r"(b) growth factor of the splitting", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_c_only.{ext}"), dpi=150)


if __name__ == "__main__":
    main(sys.argv[1])
