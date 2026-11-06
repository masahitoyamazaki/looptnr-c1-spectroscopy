from pathlib import Path
import csv, json, math, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
RATE_STARTS = (1.1, 1.5, 2.0, 2.6, 3.2)
def neldermead_1d(f, x0, step=.1, iters=4000, tol=1e-14):
    """One-parameter transcription of Otsuki's Nelder--Mead routine."""
    points = [x0, x0 + step]
    values = [f(x) for x in points]
    for _ in range(iters):
        if values[1] < values[0]:
            points.reverse(); values.reverse()
        if values[1]-values[0] <= tol*(abs(values[0])+1e-30):
            break
        center = points[0]
        reflected = 2*center-points[1]
        fr = f(reflected)
        if fr < values[0]:
            expanded = 3*center-2*points[1]
            fe = f(expanded)
            points[1], values[1] = (expanded,fe) if fe < fr else (reflected,fr)
        elif fr < values[0]:
            points[1], values[1] = reflected,fr
        else:
            contracted = .5*(center+points[1])
            fc = f(contracted)
            if fc < values[1]:
                points[1], values[1] = contracted,fc
            else:
                points[1] = .5*(points[0]+points[1]); values[1] = f(points[1])
    idx = int(np.argmin(values))
    return points[idx], values[idx]


def fit(ns, ys, k, umklapp, lattice=False):
    """Otsuki variable projection, early/main terms and RSS profile."""
    s = 2**(2-4*k)
    use_positive_lattice = lattice and abs(s-.5) > .01
    p = 4 + int(umklapp) + int(lattice) + int(use_positive_lattice)
    if len(ns)-p < 3:
        return None
    t = (ns-(ns[0]+ns[-1])/2)/10

    def solve(theta):
        exponent = theta*(ns-ns[-1])
        if np.max(exponent) > 700:
            return float("inf"), None
        cols = [np.ones(len(ns)), t, np.exp(exponent)]
        if umklapp:
            cols.append(s**(ns-ns[0]))
        if lattice:
            cols.append((-.5)**(ns-ns[0]))
        if use_positive_lattice:
            cols.append(.5**(ns-ns[0]))
        design = np.column_stack(cols)
        beta = np.linalg.lstsq(design, ys, rcond=None)[0]
        residual = ys-design@beta
        return float(residual@residual), beta

    def rss(theta):
        return solve(theta)[0]

    theta, best = None, float("inf")
    for rate0 in RATE_STARTS:
        candidate, value = neldermead_1d(rss, np.log(rate0))
        candidate, value = neldermead_1d(rss, candidate, step=.01)
        if value < best:
            theta, best = candidate, value
    if theta is None:
        return None
    residual_sum, beta = solve(theta)
    threshold = residual_sum*(1+1/(len(ns)-p))

    def edge(direction):
        inside, step = theta, .005
        while True:
            outside = inside+direction*step
            if abs(outside-theta) > 3:
                return float("nan")
            if rss(outside) > threshold:
                break
            inside, step = outside,step*1.6
        for _ in range(30):
            midpoint = (inside+outside)/2
            if rss(midpoint) > threshold:
                outside = midpoint
            else:
                inside = midpoint
        return (inside+outside)/2

    lo, hi = edge(-1),edge(1)
    return dict(r=float(np.exp(theta)), profile_lo=float(np.exp(lo)),
                profile_hi=float(np.exp(hi)), rss=residual_sum,
                rms=float(np.sqrt(residual_sum/(len(ns)-p))), n=len(ns),
                first=int(ns[0]), last=int(ns[-1]), amplitude=float(beta[2]),
                beta=[float(value) for value in beta], parameters=p)

