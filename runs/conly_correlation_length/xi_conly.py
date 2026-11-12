"""C-only runs (followup_B B7): predict the gap from the V_{0,+-2} seed via the sine-Gordon mass formula.

Seed: D(n) = x_even - x_odd of the charge-(+-1) primaries, |D| = 2 sqrt2 pi |g02| (first order, as the twisted
splitting of paper 1b), fitted with the fixed first-order rate r = 2^{1-1/(2K)} over the B7 window.
Prediction: Eq. (xi) of paper 1b with x = 1/K, p = x/(2-x): M = (2pi/L) [ |g|/(sqrt2 kappa_SG(p)) ]^{1/(2-x)}.
Measured: masses m = 2 pi x_i / L of all levels (L = 2 sqrt2 (sqrt2)^n), in units of the predicted M.
"""
import csv, glob, gzip, math, os, sys
import numpy as np
from math import gamma, pi, sqrt, sin
B7 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "additional", "data")

def kappa(p):
    return gamma(p/(p+1))/(pi*gamma(1/(p+1)))*(sqrt(pi)*gamma((p+1)/2)/(2*gamma(p/2)))**(2/(p+1))

def load(d):
    lev = {}
    with gzip.open(os.path.join(d, "levels.csv.gz"), "rt") as fh:
        for r in csv.DictReader(fh):
            lev.setdefault(int(r["step"]), []).append(r)
    st = {}
    with gzip.open(os.path.join(d, "steps.csv.gz"), "rt") as fh:
        rd = csv.reader(fh); hdr = next(rd)
        for r in rd: st[int(r[0])] = dict(zip(hdr, r))
    return lev, st

def series(lev, K):
    out = {}
    for n, rows in lev.items():
        ev = sorted(float(r["x"]) for r in rows if "(0)" in r["sector"])
        od = sorted(float(r["x"]) for r in rows if "(1)" in r["sector"])
        out[n] = (ev, od)
    return out

def main():
    print(f"{'run':18s} {'win':>6s} {'g(12)':>9s} {'xi_pred':>9s} | late steps: masses of the lowest levels / M_pred  (e: C-even, o: C-odd)")
    for d in sorted(glob.glob(os.path.join(B7, "B7_C_K*_chi*"))):
        name = os.path.basename(d); K = float(name.split("_K")[1].split("_")[0]); chi = int(name.split("chi")[1])
        lev, st = load(d); s = series(lev, K)
        x = 1/K; p = x/(2-x); r = 2**(1-1/(2*K)); ref = 1/(4*K)
        D = {}
        for n, (ev, od) in s.items():
            if len(ev) < 2 or not od: continue
            xe = min(ev[1:], key=lambda v: abs(v-ref)); xo = min(od, key=lambda v: abs(v-ref)); D[n] = xe-xo
        win = []
        for n in sorted(D):
            if n < 10: continue
            if 1e-5 < 4*K*abs(D[n]) < 1e-2: win.append(n)
            elif win: break
        if len(win) < 4:
            print(f"{name:18s} no window"); continue
        n2 = win[-1]; t = np.array([r**(n-n2) for n in win]); y = np.array([D[n] for n in win])
        A = float(t@y/(t@t))
        g12 = abs(A)*r**(12-n2)/(2*sqrt(2)*pi)
        L12 = 2*sqrt(2)*sqrt(2)**12
        M = (2*pi/L12)*(g12/(sqrt(2)*kappa(p)))**(1/(2-x))
        late = []
        for n in sorted(s)[-6:]:
            L = 2*sqrt(2)*sqrt(2)**n; ev, od = s[n]
            ms = sorted([(2*pi*v/L/M, 'e') for v in ev[1:5]] + [(2*pi*v/L/M, 'o') for v in od[:4]])
            late.append(f"n={n}: " + " ".join(f"{m:.3g}{c}" for m, c in ms[:5]) + f" c={float(st[n]['central_charge']) if n in st and 'central_charge' in st[n] else float('nan'):.3f}")
        print(f"{name:18s} {win[0]}-{n2:<3d} {g12:9.2e} {1/M:9.2e} | p={p:.3f}")
        for l in late: print("      ", l)

if __name__ == "__main__":
    main()
