#!/usr/bin/env python3
"""Iterative refinement of the one-shot counterterm (2026-11-12, Yamazaki).
Starting from the one-shot prediction mu_pred and the twisted splitting delta_pi(n_cal) of its run, adjust mu so
that delta_pi(n_cal) vanishes in the actual runs: first a Newton step with the calibrated slope s(n_cal; eps/4),
then secant steps, each a short run to n_cal with tw_run.jl (same LoopTNR parameters and readout).  Stop when
|delta| < TOL * |delta(mu=0)| or after MAXIT runs; then one full run with the best mu to the breakdown.
Usage: python3 iterate.py GROUP     (groups below; writes out/I_<group>_it<k>.csv, out/I_<group>_final.csv, iterate_<group>.log)
"""
import os, subprocess, sys, math
HERE = os.path.dirname(os.path.abspath(__file__))
# group: K, chi, n_cal, mu_pred, delta_pi(n_cal) of the mu_pred run, calibrated slope, delta_pi(n_cal) at mu = 0
G = {
    "K1.25_chi40": (1.25, 40, 20, 6.002397898380329e-6, 1.8549894477470907e-5, -317.50045240801006, 1.906e-3),
    "K1.5_chi24": (1.5, 24, 24, -1.0151824567950324e-4, 7.905461810285286e-4, -150.79989744320622, -1.5309092065871e-2),
    "K1.5_chi32": (1.5, 32, 24, 3.4613384606426556e-5, 4.1536278613596433e-4, -146.42169985491526, 5.07e-3),
    "K1.5_chi40": (1.5, 40, 24, 2.1259331404370927e-5, 2.7894047609750006e-4, -149.001021342829, 3.1635966567961638e-3),
}
TOL, MAXIT = 1e-3, 3

def run(K, chi, mu, steps, stop, out):
    env = dict(os.environ, K=str(K), CHI=str(chi), H=repr(mu), STEPS=str(steps), STOP=str(stop), PERTURB="0", SEED="1", FORCE2="0", OUT=out)
    subprocess.run([os.path.expanduser("~/.juliaup/bin/julia"), f"--project={os.path.expanduser('~/tnr_cpu')}", "-t", "1",
                    os.path.join(HERE, "tw_run.jl")], env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def delta_at(path, n):
    for line in open(path):
        if line.startswith(f"{n},"):
            return float(line.strip().split(",")[2])
    raise RuntimeError(f"step {n} missing in {path}")

def n_e(path):
    rows = [l.strip().split(",") for l in open(path) if l[0].isdigit()]
    for n, X, _ in rows:
        if int(n) >= 6 and abs(float(X)) > 0.1:
            return int(n) - 1
    return None

def main(g):
    K, chi, ncal, mu_p, d_p, s, d0 = G[g]
    log = open(os.path.join(HERE, f"iterate_{g}.log"), "a")
    say = lambda *a: (print(*a, file=log, flush=True))
    say(f"# {g}: n_cal={ncal} mu_pred={mu_p:.6e} delta(mu_pred)={d_p:.3e} ({d_p/d0:+.4f} of delta(0)) slope={s:.2f}")
    pts = [(mu_p, d_p)]
    mu = mu_p - d_p / s
    for k in range(1, MAXIT + 1):
        out = os.path.join(HERE, "out", f"I_{g}_it{k}.csv")
        run(K, chi, mu, ncal, 0, out)
        d = delta_at(out, ncal); pts.append((mu, d))
        say(f"it{k}: mu={mu:.8e} delta={d:.4e} ({d/d0:+.5f} of delta(0))")
        if abs(d) < TOL * abs(d0):
            break
        (m1, e1), (m2, e2) = pts[-2], pts[-1]
        mu = m2 - e2 * (m2 - m1) / (e2 - e1) if e2 != e1 else m2 - e2 / s
    best = min(pts, key=lambda p: abs(p[1]))
    out = os.path.join(HERE, "out", f"I_{g}_final.csv")
    run(K, chi, best[0], 120, 1, out)
    say(f"final: mu={best[0]:.8e} (delta/delta0={best[1]/d0:+.5f}) n_e={n_e(out)}")

if __name__ == "__main__":
    main(sys.argv[1])
