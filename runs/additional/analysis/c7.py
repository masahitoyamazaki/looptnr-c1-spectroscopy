"""Runs C7: single-step replacement along the reference trajectory (O(2), K = 1.5, chi = 40).

Usage:  python3 c7.py          (reads ../data/C7_K1.5_n<n>_<variant>/ written by ../c7_one.sh with STEP_OVERRIDE)
Each run continues from the reference checkpoint step_{n-1} with only step n modified -- chi64: truncated to
chi = 64; s80: 80 loop-optimization sweeps; s0: no loop optimization (initial decomposition and projectors only) --
and all other steps at the default settings.  Prints g_d^2(12) relative to the reference run and n_e.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from b1_b2 import seed, BASE   # noqa: E402

DATA = os.path.join(HERE, "..", "data")
ref = seed(os.path.join(BASE, "CU1_K1.5000_chi40"), 1.5)
print(f"reference: g2(12) = {ref['g2']:.3e}, n_e = {ref['ne']}")
print(" n | chi64            | s80              | s0 (no loop optimization)")
for n in range(5, 12):
    cells = []
    for v in ("chi64", "s80", "s0"):
        f = seed(os.path.join(DATA, f"C7_K1.5_n{n}_{v}"), 1.5)
        cells.append(f"x{f['g2'] / ref['g2']:5.2f} (n_e={f['ne']})" if f else "no window")
    print(f"{n:2d} | " + " | ".join(cells))
