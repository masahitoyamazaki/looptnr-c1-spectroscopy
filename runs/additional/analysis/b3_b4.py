"""Analysis of runs B3 (branches from common checkpoints) and B4 (loop-optimization convergence).

Usage:  python3 b3_b4.py OUT_DIR      (B3_*/ and B4_*/ with levels.csv, steps.csv from the workstation)
Reference runs (same environment, from scratch) are taken from ../../cpu_baseline_20261106/data.
For each run: g_d^2(n=12), the end of the weak-coupling window n_e and r/2^(2-K) as in b1_b2.py;
for B4 also the mean final loop-optimization cost over n = 6-20.
B3 "same" branches must reproduce the reference exactly (determinism check).
"""
import csv, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from b1_b2 import seed, BASE          # noqa: E402
from analyze import load              # noqa: E402


def ref(K, chi):
    return seed(os.path.join(BASE, f"CU1_K{K:.4f}_chi{chi}"), K)


def mean_cost(d, n1=6, n2=20):
    _, steps = load(d)
    c = [abs(float(steps[n]["final_cost"])) for n in range(n1, n2 + 1) if n in steps]
    return float(np.mean(c)) if c else float("nan")


def main(out):
    rows = []
    print("== B3: branches from the chi=40 checkpoint at step n0")
    for K in (1.0, 1.5):
        r40, r32, r48 = ref(K, 40), ref(K, 32), ref(K, 48)
        print(f" K={K}: references from scratch: chi=40 g2={r40['g2']:.2e} n_e={r40['ne']}; "
              f"chi=32 g2={r32['g2']:.2e} n_e={r32['ne']}; chi=48 g2={r48['g2']:.2e} n_e={r48['ne']}")
        for n0 in (6, 10):
            for var in ("same", "chi32", "chi48", "s80c9", "ef12"):
                d = os.path.join(out, f"B3_K{K}_n{n0}_{var}")
                if not os.path.isdir(d):
                    continue
                f = seed(d, K)
                print(f"   n0={n0:2d} {var:6s}: g2={f['g2']:.2e} (x{f['g2'] / r40['g2']:.2f} of chi=40), n_e={f['ne']}, "
                      f"r/2^(2-K)={f['r'] / 2 ** (2 - K):.4f}")
                rows.append(dict(set="B3", K=K, chi=None, n0=n0, variant=var, g2_n12=f["g2"], n_e=f["ne"],
                                 r_over_pred=f["r"] / 2 ** (2 - K), g2_ref40=r40["g2"]))
        # determinism: the "same" branch against the reference run
        d = os.path.join(out, f"B3_K{K}_n6_same")
        if os.path.isdir(d):
            a, _ = load(d); b, _ = load(os.path.join(BASE, f"CU1_K{K:.4f}_chi40"))
            diffs = [abs(float(x["x"]) - float(y["x"])) for n in a if n in b
                     for x, y in zip(sorted(a[n], key=lambda r: (r["sector"], r["rank"])), sorted(b[n], key=lambda r: (r["sector"], r["rank"])))]
            print(f"   determinism (same vs reference): max |dx| = {max(diffs):.1e} over {len(diffs)} levels")
    print("== B4: loop-optimization convergence")
    for K, chi in ((1.5, 24), (1.5, 40), (1.0, 40)):
        r = ref(K, chi)
        rc = mean_cost(os.path.join(BASE, f"CU1_K{K:.4f}_chi{chi}"))
        print(f" K={K} chi={chi}: default (20 sweeps, 1e-9): g2={r['g2']:.2e} n_e={r['ne']} mean cost {rc:.1e}")
        for var in ("s80c12", "s320c12"):
            d = os.path.join(out, f"B4_K{K}_chi{chi}_{var}")
            if not os.path.isdir(d):
                continue
            f = seed(d, K)
            c = mean_cost(d)
            if f is None:
                print(f"   {var}: not yet past |X| = 0.1 (running); mean cost {c:.1e}")
                continue
            print(f"   {var}: g2={f['g2']:.2e} (x{f['g2'] / r['g2']:.2f}), n_e={f['ne']}, r/2^(2-K)={f['r'] / 2 ** (2 - K):.4f}, mean cost {c:.1e}")
            rows.append(dict(set="B4", K=K, chi=chi, n0=None, variant=var, g2_n12=f["g2"], n_e=f["ne"],
                             r_over_pred=f["r"] / 2 ** (2 - K), g2_ref40=r["g2"]))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(HERE, "b3_b4.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1])
