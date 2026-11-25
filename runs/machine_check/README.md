# machine_check_20261124: round-off differences between machines (paper 1b, App. A and Sec. VIII)

The standard LoopTNR run (O(2), K = 1, chi = 35, TNRKit 0.7.0, TensorKit 0.16.5, default loop parameters) was repeated with the same script on two machines: a Mac (Julia 1.10.4) and a Linux machine (Julia 1.11.9).

- `xi_machine.jl` runs to |X| > 3 and records X(n), x01, g_d^pi(n), and the predicted xi at every step. It uses `readout_common.jl`, the readout of the paper 1b seed experiments.
- `out/ximachine_{mac,a6000}.csv` holds the two runs.
- `compare.py` gives the seed ratio and the xi ratio from the steps at which |X| crosses 0.1, 0.3, 1, and 2.

Result: the seeds differ by 10% from n = 8 on. The measured xi differs by 11% (1.110–1.113) and the predicted xi by the same factor (1.111). n_e = 29 on both machines.
