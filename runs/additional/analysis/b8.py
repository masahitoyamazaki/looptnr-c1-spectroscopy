"""Runs B8 at the BKT point K = 1/2 (Delta_6v = -1; O(2) LoopTNR, chi = 24-64).

Usage:  python3 b8.py            (reads ../data/B8_K0.5_chi*/levels.csv.gz)

At K = 1/2 the four primaries of dimension 1/2 form the SU(2)_1 triplet [(0,+-1) and the C-odd e = +-1
combination] and singlet [C-even e = +-1, i.e. V_d].  The marginally irrelevant umklapp J.Jbar gives
    x_t = 1/2 - g/4,  x_s = 1/2 + 3g/4,   dg/dl = -g^2 - g^3/2,   l = ln L,
so 1/g = ln(L/L0) + (1/2) ln ln(L/L0) + O(1) [Affleck et al. 1989; Affleck 1998; Barzykin-Affleck 1999],
and with ln L = n ln sqrt(2) + const the slope of 4/g = 1/(1/2 - x_t) per step is 4 ln sqrt(2) (1 + g/2).
We print g_t = 4(1/2 - x_t), g_s = (4/3)(x_s - 1/2), the log-free combination (x_s + 3 x_t)/4 - 1/2, and
the local slope of 1/(1/2 - x_t) compared with 4 ln sqrt(2) (1 + g_t/2).  Writes fig_bkt.pdf/png.
"""
import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "main", "analysis"))
from analyze import load   # noqa: E402

DATA = os.path.join(HERE, "..", "data")
L2 = math.log(math.sqrt(2))


def series(chi):
    lev, steps = load(os.path.join(DATA, f"B8_K0.5_chi{chi}"))
    out = {}
    for n, rows in lev.items():
        ch = sorted(float(r["x"]) for r in rows if "(1, 2)" in r["sector"])
        od = sorted(float(r["x"]) for r in rows if "(0, 1)" in r["sector"])
        ev = sorted(float(r["x"]) for r in rows if "(0, 0)" in r["sector"])
        if not ch or not od or len(ev) < 2:
            continue
        xt = 0.5 * (ch[0] + od[0])
        out[n] = dict(xt=xt, split_t=ch[0] - od[0], xs=ev[1], c=float(steps[n]["central_charge"]))
    return out


def main():
    S = {chi: series(chi) for chi in (24, 32, 40, 48, 64)}
    print(" chi   n   x_t       x_s      g_t     g_s    (xs+3xt)/4-1/2  triplet split")
    for chi in (40, 64):
        for n in range(8, 27, 2):
            r = S[chi].get(n)
            if r:
                gt, gs = 4 * (0.5 - r["xt"]), 4 / 3 * (r["xs"] - 0.5)
                print(f" {chi:3d} {n:3d}  {r['xt']:.5f}  {r['xs']:.5f}  {gt:.4f}  {gs:.4f}  {(r['xs'] + 3 * r['xt']) / 4 - 0.5:+.2e}  {r['split_t']:+.1e}")
    for chi in (40, 48, 64):
        ns = np.array([n for n in sorted(S[chi]) if 8 <= n <= 22], float)
        y = np.array([1 / (0.5 - S[chi][int(n)]["xt"]) for n in ns])
        p = np.polyfit(ns, y, 1)
        gmid = 4 * (0.5 - S[chi][15]["xt"])
        print(f" chi={chi}: slope of 1/(1/2-x_t) over n=8-22: {p[0]:.3f} per step; leading 4 ln sqrt2 = {4 * L2:.3f}; "
              f"with ln ln term 4 ln sqrt2 (1+g/2) = {4 * L2 * (1 + gmid / 2):.3f} (g = {gmid:.3f} at n = 15)")
    figure(S)


def figure(S):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts (journals reject Type 3)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for chi, col in ((40, "C0"), (64, "C3")):
        ns = [n for n in sorted(S[chi]) if 4 <= n <= 26]
        ax[0].plot(ns, [S[chi][n]["xt"] for n in ns], "o", ms=2.5, color=col, label=rf"triplet, $\chi={chi}$")
        ax[0].plot(ns, [S[chi][n]["xs"] for n in ns], "s", ms=2.5, mfc="none", color=col, label=rf"singlet, $\chi={chi}$")
        ax[0].plot(ns, [(S[chi][n]["xs"] + 3 * S[chi][n]["xt"]) / 4 for n in ns], "-", lw=1, color=col,
                   label=rf"$(x_s+3x_t)/4$, $\chi={chi}$")
    ax[0].axhline(0.5, color="k", lw=0.6, ls="--")
    ax[0].set_ylim(0.34, 0.75); ax[0].set_xlabel("RG step $n$"); ax[0].set_ylabel("$x$")
    ax[0].legend(fontsize=5.2, ncol=2, frameon=False, loc="lower left")
    ax[0].set_title(r"(a) $K=1/2$: $SU(2)_1$ triplet and singlet", fontsize=8.5)
    for chi, col in ((40, "C0"), (64, "C3")):
        ns = np.array([n for n in sorted(S[chi]) if 6 <= n <= 24], float)
        ax[1].plot(ns, [1 / (0.5 - S[chi][int(n)]["xt"]) for n in ns], "o", ms=2.5, color=col, label=rf"$1/(1/2-x_t)$, $\chi={chi}$")
        ax[1].plot(ns, [3 / (S[chi][int(n)]["xs"] - 0.5) for n in ns], "s", ms=2.5, mfc="none", color=col,
                   label=rf"$3/(x_s-1/2)$, $\chi={chi}$")
    n0 = 15; y0 = 1 / (0.5 - S[64][n0]["xt"]); g = 4 * (0.5 - S[64][n0]["xt"])
    nn = np.arange(6, 25)
    ax[1].plot(nn, y0 + 4 * L2 * (nn - n0), "k:", lw=0.9, label=r"slope $4\ln\sqrt{2}$")
    ax[1].plot(nn, y0 + 4 * L2 * (1 + g / 2) * (nn - n0), "k--", lw=0.9, label=r"slope $4\ln\sqrt{2}\,(1+g/2)$")
    ax[1].set_xlabel("RG step $n$  ($\\propto\\ln L$)"); ax[1].legend(fontsize=5.5, frameon=False)
    ax[1].set_title(r"(b) logarithmic approach", fontsize=8.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_bkt.{ext}"), dpi=200)


if __name__ == "__main__":
    main()
