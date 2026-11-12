# followup_C_20261110: sign of the seed at chi = 48, 64 and readout control of the wide tube

GPU workstation (CPU jobs, Julia 1.13.3, TNRKit 0.7.0), 2026-11-10/11. Scripts in `scripts/` (copies of
followup_B `run2.jl`, `common.jl`, `worker.sh`; `tasks.tsv`; `wide_readout_trunc.jl`). Checkpoints were not kept.

## 1. Sign of the seed (16 runs, reserved as `yamazaki_stagger_ab_{p,m}{mu}`)
O(2), K = 1, 1.5, chi = 48, 64, STAGGER_AB = +-mu0, +-2 mu0 (a/b alternation), 80 steps or stop at |c-1|>0.9.
mu0 = 3e-6, 1.5e-6 (K=1, chi=48, 64); 4e-5, 1e-5 (K=1.5). mu = 0 runs: main/data.
`analysis/sign_test.py` -> `analysis/sign_test.csv`.

| K | chi | g_seed | kappa | mu* |
|---|---|---|---|---|
| 1.0 | 48 | -2.27e-5 | 13.67 | +1.66e-6 |
| 1.0 | 64 | -1.01e-5 | 14.17 | +7.2e-7 |
| 1.5 | 48 | -3.82e-5 | 2.11 | +1.81e-5 |
| 1.5 | 64 | -1.10e-5 | 2.02 | +5.4e-6 |

Same sign as chi = 32, 40 (followup_B b9.csv); chi = 24 at K = 1.5 remains the only opposite case.
kappa agrees with chi = 24-40 within 5%. mu0 turned out to be ~2 mu*, so these runs give no compensation.

## 2. Wide-tube readout with relaxed readout truncation (4 readouts)
`wide_readout_trunc.jl` with CHI_READ = 2 chi on the cpu_baseline checkpoints (K = 1, 1.5; chi = 32, 40).
`analysis/readout_control.py` -> `analysis/readout_control.csv` (compares with followup_B W1, CHI_READ = chi).
5^{2-K} factor changes by <= 1.4% (geometric); the amplitude ratio at K = 1.5 moves from 0.963/0.969 to
0.995/1.000, i.e., the few-percent wide-tube deviation is due to the readout truncation.

Run: `python3 analysis/sign_test.py; python3 analysis/readout_control.py` (needs numpy, scipy).
