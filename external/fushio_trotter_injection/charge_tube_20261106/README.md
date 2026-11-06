# Note 13: charged readout on the larger spatial tube

This bundle reuses the note 11 checkpoints at even RG steps 0–16, with
`chi_loop=32`, `chi_stack=64`, `dt=0.125`, `beta_cell=4`, `K≈1.5`, and
`h=0,0.01,0.02`. It does not advance RG or modify tensor checkpoints.

The tube is `[1,4,1]`. After the same even-step orientation correction as note
11, its physical circumference is `8*2^(step/2)` and its temporal advance is
`beta_cell*2^(step/2)`. The modular parameter is `(1+tau0)/4`.

Run from the repository root with the recorded Julia project:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --startup-file=no --threads=1 --project=Fushio/vendor/TNRKit.jl Fushio/docs/six_vertex/dimerization_injection/support/charge_tube_20261106/measure.jl
python3 Fushio/docs/six_vertex/dimerization_injection/support/charge_tube_20261106/analyze.py
python3 Fushio/docs/six_vertex/dimerization_injection/support/charge_tube_20261106/fit_growth.py
python3 Fushio/docs/six_vertex/dimerization_injection/support/charge_tube_20261106/write_report.py
python3 Fushio/docs/six_vertex/dimerization_injection/support/charge_tube_20261106/finalize.py
python3 Fushio/docs/six_vertex/dimerization_injection/support/charge_tube_20261106/validate.py
```

Completed files are skipped. A `.progress` file holds fully measured sectors;
resuming checks the checkpoint hash and reuses those sectors. Do not run two
workers on the same job. The recorded production workers use the disjoint
`control/jobs0.tsv`, `jobs1.tsv`, and `jobs2.tsv` files for RG6–16,
and `control/early_jobs0.tsv`, `early_jobs1.tsv`, `early_jobs2.tsv` for RG0/2/4.
The default `control/jobs.tsv` includes all 27 points; completed points are skipped.

`control/inputs.json` records all 27 input checkpoints and their SHA-256.
`control/source_hashes.json` records the numerical source files.
`validate_geometry.jl` compares the initial physical tube with independent
finite-Trotter spin-chain transfer matrices. `control/slicing_validation.toml`
records an exact-contraction comparison on random CU1 tensors.

`data/` contains per-checkpoint eigenvalues, phases, residuals and two-start
agreement. `analysis.json` and `observables.csv` contain the signed,
background-subtracted estimates. `control/local_cft_copies.json` maps the
readouts into their original checkpoint branches under `CFT/tube=1_4_1/`.

The interrupted initial reader and contraction benchmarks are retained under
`control/` for provenance. They did not change RG or the transfer operator.
The saved vacuum sector of the first initial reader used Krylov dimension 24;
the main production reader requests only the leading eigenvalue per sector.
RG8 onward checks Arnoldi convergence eagerly; the eigenvalue was compared with
the original solver on an actual chi32 sector (`control/eager_validation.toml`).
No Google Drive upload is part of this task.

The early extension keeps the old local CFT copies intact and writes the full
RG0–16 series to `measurement=charged_note13_early_20261106/`. At RG0 the
small tube has no m=3 sector; its m3 and X are stored as null, without extrapolation.
The lowest charged-level figure plots the signed difference x1 - 1/(4*K0) on
a logarithmic vertical axis, against linear RG steps. All measured differences
are positive. The CSV retains raw x1 and adds x1_minus_cft.

Growth fits use saved background-subtracted estimates only. `fit_growth.py`
fits log(G) with equal weights, separately for each reader; primary windows
start at RG6 and end at the last contiguous matched weak-gate point.
`growth_fits.json` pins its input hash, source hash, selection rule, all window
fits, and local rates. Reader/window spreads are not statistical confidence intervals.
