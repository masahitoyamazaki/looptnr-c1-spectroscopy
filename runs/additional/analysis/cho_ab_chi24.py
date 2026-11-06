"""K.C.'s alternating a/b runs (O(2), K = 1.5, chi = 24, h = +-1e-3, +-2e-3, 24 steps; 2026-11-02, paused,
marked preliminary) read with the same growth fit as b1_b2.py, from the stored lowest charged level only.

Usage:  python3 cho_ab_chi24.py
Input:  Cho/research_notes/results/loop_source_calibration_20261102/saved_inventory.csv (x1 = x_{0,1}(n)); not modified.
The window is n = 12 to the step before |X| > 0.1 or the last stored step (24).  The pairs +-h give
c = [g(+h) + |g(-h)|]/(2h) and g_seed = [g(+h) - |g(-h)|]/2 with g(+h) > 0 (sign relative to h).
"""
import csv, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "main", "analysis"))
from analyze import fit_growth, ALPHA   # noqa: E402
INV = os.path.join(HERE, "..", "..", "..", "external", "cho_ab_chi24", "saved_inventory.csv")
K = 1.5
a01 = float({round(float(r["K"]), 4): r for r in csv.DictReader(open(ALPHA))}[1.5]["m1"])
runs = {}
for r in csv.DictReader(open(INV)):
    runs.setdefault(float(r["h"]), {})[int(r["step"])] = float(r["x1"])
g = {}
for h in sorted(runs):
    X = {n: 1 - 4 * K * x for n, x in runs[h].items()}
    out = [m for m in sorted(X) if m >= 12 and abs(X[m]) > 0.1]
    ne = out[0] - 1 if out else max(X)
    n = np.arange(12, ne + 1, dtype=float)
    _, lr, c, _ = fit_growth(n, np.array([X[int(m)] for m in n]), ne, False, K)
    g[h] = math.sqrt(max(-c[2] * math.exp(lr * (12 - ne)) / (4 * K * a01), 0))
    print(f"h={h:+.3f}: window 12-{ne}, r/2^(2-K)={math.exp(lr) / 2 ** (2 - K):.4f}, g_d^2(12)={g[h] ** 2:.3e}")
for a in (1e-3, 2e-3):
    print(f"|h|={a:.0e}: c = {(g[a] + g[-a]) / (2 * a):.2f}, g_seed = {(g[a] - g[-a]) / 2:+.2e}")
