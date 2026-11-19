# looptnr-c1-spectroscopy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24361530.svg)](https://doi.org/10.5281/zenodo.24361530)

Archived at Zenodo: [10.5281/zenodo.24361530](https://doi.org/10.5281/zenodo.24361530) (all versions; v1.0: [10.5281/zenodo.24361531](https://doi.org/10.5281/zenodo.24361531)).

Data and code for the two papers

> **Paper I.** A. Otsuki, Y. Yamamoto, K. Cho, R. Fushio, and M. Yamazaki,
> *Symmetry-resolved spectroscopy of truncation-induced perturbations in loop tensor network renormalization of the six-vertex model*
> (arXiv: to be added).
>
> **Paper II.** R. Fushio, K. Cho, Y. Yamamoto, A. Otsuki, and M. Yamazaki,
> *Truncation seed in loop tensor network renormalization: correlation length, signed readout, and compensation*
> (arXiv: to be added; author list provisional).

Paper I identifies the perturbations by which finite-bond-dimension truncation errors of loop tensor network
renormalization (LoopTNR) leave the critical line of the six-vertex model, using the transfer-matrix spectra of the
coarse-grained tensors and conformal perturbation theory.  Paper II shows that the seed of the dominant perturbation
predicts the correlation length of a run, reads it with its sign, and compensates it.  This repository contains the
transfer-matrix spectra of all runs used in the two papers, the scripts that produced them, and the analysis scripts
that produce the figures, tables, and quoted numbers.  The coarse-grained tensors themselves (checkpoints) are not
included because of their size; they are available from the authors upon request.

## Contents

| Folder | Contents |
|---|---|
| `environment/` | Julia environment of all runs (`Project.toml`, `Manifest.toml`; TNRKit 0.7.0, TensorKit 0.16.5, Julia 1.13.3) |
| `runs/main/` | Main runs: $O(2)$ and $U(1)$ LoopTNR, $K=0.75$–$2.5$, $\chi=16$–$64$, up to 80 RG steps (`run.jl`, `common.jl`) |
| `runs/main/data/` | Spectra of the main runs, one folder per run: `levels.csv.gz` (step, sector, rank, scaling dimension, eigenvalue phase, ...) and `steps.csv.gz` (central charge, timings, ...) |
| `runs/main/analysis/` | Fits, tables, and figures of Paper I, Secs. IV–V (`analyze.py`, `drift.py`, `drift_table.py`, `induced_drift.py`, `synthetic_flow.py`, `paper_figures.py`, `figures.py`, `checks.py`, `bootstrap.py`); second-order coefficients `sixvertex_g2_alpha.csv` (Table VIII) |
| `runs/additional/` | Additional runs with the same environment (`run2.jl` with options; `readout.jl`, `wide_readout.jl` for re-reading saved tensors) |
| `runs/additional/data/` | Spectra of the additional runs; folder prefixes as listed in the run appendices of the papers (B1–B9d, C7, JJ, W1) |
| `runs/additional/sv/out/` | Singular values at the truncations (A2) |
| `runs/additional/analysis/` | Analyses of both papers (one script per data set; see the docstring of each script) |
| `external/` | Data from separate runs by individual authors (see below) |
| `earlier/` | Independent runs with a development version of the code and their analysis (Paper I, Appendix C 1, and the auxiliary checks of the current) |
| `runs/signed_readout/` | Paper II: runs with the $\pi$-twisted readout, one-shot and refined compensation (`tw_run.jl`, `iterate.py`), exact diagonalization with a twist (`ed/`) |
| `runs/step_injection/` | Paper II: per-step injection of the seed, steps 5–11 replaced by $\chi=64$ or 80 sweeps, read with the twisted readout (`si_run.jl`, `analyze.py`) |
| `runs/xi_sine_gordon/`, `runs/seed_refit/`, `runs/coupling_consistency/` | Paper II: correlation length versus the sine-Gordon prediction, refit of the compensation scans, three readouts of the coupling |
| `runs/seed_refit/protocol_factor.py`, `protocol_factor.csv` | Paper I, Appendix C: seed of the main fits versus the fixed-rate protocol of Paper II |
| `runs/sign_and_wide_readout/`, `runs/width_check/` | Paper II: sign of the seed at $\chi=48$, 64; readout with wider cylinders |
| `runs/conly_correlation_length/` | Paper II, appendix on the $C$-only runs: first-breather gap predicted from the $V_{0,\pm2}$ seed |
| `runs/compensation_drift/` | Paper II: drift of $K_n$ with and without compensation |
| `runs/midrg_compensation/` | Paper II, Sec. VI: corrections during the RG with four constraints (K. Cho's projected-gradient code in `cho_src/`), $K=0.5$, $1$, $1.5$, $\chi=24$--$40$, up to step 160 |
| `runs/sweep80/` | Paper II, App. B: integer $\chi$ scan at $K=1.5$ ($\chi=24$--$48$) repeated with 80 loop-optimization sweeps; `analyze.py` compares seeds, breakdown steps, and $\xi$ with the default scan |
| `runs/window_extrapolation/` | Paper I, Appendix C 2: dependence on the fit window |

## Reproducing the figures and tables

Requirements: Python 3.9 or later with `numpy`, `scipy`, and `matplotlib` (and `physics-tenpy` for the DMRG check).
From the repository root:

```sh
sh run_all.sh
```

which runs, in this order,

| Script (folder `runs/main/analysis` unless noted) | Produces |
|---|---|
| `analyze.py ../data` | `o2_fits.csv`, `u1_fits.csv` (growth fits; Paper I, Tables IV, VI, and VII) |
| `drift.py ../data` | $K_n$ and $\eta$ (Paper I, Sec. IV F; needed by `paper_figures.py`) |
| `drift_table.py` | Direct and induced shifts of $K$ (Paper I, Table V and App. D) |
| `induced_drift.py` | Shift of $K_n$ induced by the dimerization (Paper I, App. D) |
| `synthetic_flow.py` | Synthetic flow with a running $K$: bias of the rate and amplitude ratio (Paper I, Sec. IV B and App. D) |
| `paper_figures.py` | Figures of the $O(2)$ fit terms, growth rate, amplitude ratio, level repulsion at $K=1$, $U(1)$ flows and growth factor, drift; Tables IV and VI (`paper_tables.tex`) |
| `figures.py ../data` | Figures of the eigenvalue phases ($U(1)$) and of the seed versus $\chi$ |
| `bootstrap.py` | Residual-bootstrap uncertainties (Paper I, Sec. IV B) |
| `runs/additional/analysis/b7.py ../data` | Figure of the $C$-only splitting (Paper I, Sec. V C) |
| `runs/additional/analysis/b8.py` | Figure of the BKT point $K=1/2$ (Paper I, Sec. IV E) |
| `runs/additional/analysis/b9.py ../data` | Figure and Table II of the compensation (Paper II, Sec. IV B) |
| `runs/additional/analysis/jj.py`, `wide_tube.py`, ... | Numbers quoted in the text (marginal injection, wider tube, single-step replacements, singular values, ...); the branch seeds of Paper II, Table V, with the present protocol are in `runs/seed_refit/` (`b3_b4.csv` uses the earlier free-rate protocol) |
| `runs/additional/analysis/xxz_alpha_check.py` | Exact-diagonalization check of the amplitude-ratio coefficients (Paper I, Appendix B) |

Two figures of Paper I are based on the independent runs, included in `earlier/` with their own scripts
(`earlier/B_1_u1_tube_charge_checks`, auxiliary checks of the current; `earlier/3_2_growth_rate`, dense scan in $K$).  Run `python3 scripts/reproduce.py`
inside each folder (Python 3.10 or later; `pip install -r requirements.txt`); it re-analyzes the stored spectra and
compares the results with the stored reference values.  In these folders, local paths and links to the authors'
shared drive were redacted for publication and the input hashes in `provenance.json` were recomputed accordingly.

New RG runs are started with, e.g.,

```sh
julia --project=environment runs/main/run.jl CU1 1.5 40 80 OUTDIR
```

(see the header of `run.jl` and `run2.jl` for the arguments and options).

## External data

`external/` contains data produced by individual authors with their own scripts, of which the paper uses only the
stored spectra:

- `cho_ab_chi24/`: alternating $a$/$b$ runs at $K=1.5$, $\chi=24$ (K. Cho; Paper II)
- `yamamoto_conly_scan/`: $C$-only scan at $\chi=16$ over 17 values of $K$ and its fit code (Y. Yamamoto; Paper I, Sec. V C)
- `fushio_trotter_injection/`: Trotter injection of the dimerization read with a wider tube (R. Fushio; Paper II)
- `otsuki_sg_xi/`: per-run tables of the correlation-length analysis (window fits, measured and predicted $\xi$, levels per step) and the levels of the saved runs used for the mass ratios (A. Otsuki; Paper II, Sec. III), read by `runs/xi_sine_gordon/`
- `yamamoto_xi_report10/`: the second, independent analysis of the correlation length and the BKT point (Y. Yamamoto; Paper II, Sec. III and Apps. D and E), read by `runs/xi_sine_gordon/`
- `cho_midrg_interval/`: corrections during the RG at the BKT point, every four versus every two steps (K. Cho; Paper II, Sec. VI)
- `fushio_kn_drift/`: slopes of the drift of $K$ from three level combinations, $K_n$, $x_e/x_J$, and $x_J/4x_{0,1}$, read from the stored spectra of `runs/main/` (R. Fushio; Paper I, Sec. IV F and App. D)
- `otsuki_ff_kappa/`: exact lowest charged level of the dimerized XX chain (free fermions, $K=1$) and the coefficient $\kappa=0.2345$ of the fourth-order shift, and the fits with a $\kappa X_g^2$ term of the rate deficit for $K\le1$, with their scripts (A. Otsuki; Paper I, Sec. IV B and App. D)

The $C$-only runs with a transverse field (K. Cho; Paper I, Sec. V C) are in `runs/additional/data/cho_transverse_field_20261024/` (`result.toml` and `config.toml` of each run) and are analyzed by `runs/additional/analysis/v01_signature.py`.

## Licenses

Code: MIT License (`LICENSE`).  Data (spectra and derived tables): Creative Commons Attribution 4.0 International
(`LICENSE-DATA`).  If you use the data, please cite the papers and the Zenodo record (doi:10.5281/zenodo.24361530).

## Use of AI tools

Scripts in this repository were written with the help of AI coding assistants (Anthropic Claude Opus 5.5 via
Claude Code; OpenAI Codex with GPT-6 Astra, GPT-6 Sol, and GPT-6.1 Sol), as stated in the papers; all results were
checked by the authors.
