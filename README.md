# looptnr-c1-spectroscopy

Data and code for the paper

> A. Otsuki, R. Fushio, K. Cho, Y. Yamamoto, and M. Yamazaki,
> *Symmetry-resolved spectroscopy of truncation-induced perturbations in loop tensor network renormalization at c = 1*
> (arXiv: to be added).

The paper studies how finite-bond-dimension truncation errors of loop tensor network renormalization (LoopTNR) grow
on the critical line of the six-vertex model, using the transfer-matrix spectra of the coarse-grained tensors and
conformal perturbation theory.  This repository contains the transfer-matrix spectra of all runs used in the paper,
the scripts that produced them, and the analysis scripts that produce the figures, tables, and quoted numbers.
The coarse-grained tensors themselves (checkpoints) are not included because of their size; they are available
from the authors upon request.

## Contents

| Folder | Contents |
|---|---|
| `environment/` | Julia environment of all runs (`Project.toml`, `Manifest.toml`; TNRKit 0.7.0, TensorKit 0.16.5, Julia 1.13.3) |
| `runs/main/` | Main runs: $O(2)$ and $U(1)$ LoopTNR, $K=0.75$–$2.5$, $\chi=16$–$64$, up to 80 RG steps (`run.jl`, `common.jl`) |
| `runs/main/data/` | Spectra of the main runs, one folder per run: `levels.csv.gz` (step, sector, rank, scaling dimension, eigenvalue phase, ...) and `steps.csv.gz` (central charge, timings, ...) |
| `runs/main/analysis/` | Fits, tables, and figures of Secs. IV–V (`analyze.py`, `drift.py`, `paper_figures.py`, `figures.py`, `checks.py`, `bootstrap.py`); second-order coefficients `sixvertex_g2_alpha.csv` (Table VII) |
| `runs/additional/` | Additional runs with the same environment (`run2.jl` with options; `readout.jl`, `wide_readout.jl` for re-reading saved tensors) |
| `runs/additional/data/` | Spectra of the additional runs; folder prefixes as listed in Appendix C of the paper (B1–B9d, C7, JJ, W1) |
| `runs/additional/sv/out/` | Singular values at the truncations (A2) |
| `runs/additional/analysis/` | Analyses of Secs. IV–VI and the appendices (one script per data set; see the docstring of each script) |
| `external/` | Data from separate runs by individual authors used in Secs. V C and VI (see below) |
| `earlier/` | Earlier runs and their analysis for Appendices D and E |

## Reproducing the figures and tables

Requirements: Python 3.9 or later with `numpy`, `scipy`, and `matplotlib` (and `physics-tenpy` for the DMRG check).
From the repository root:

```sh
sh run_all.sh
```

which runs, in this order,

| Script (folder `runs/main/analysis` unless noted) | Produces |
|---|---|
| `analyze.py ../data` | `o2_fits.csv`, `u1_fits.csv` (growth fits; Table IV) |
| `drift.py ../data` | $K_n$ and $\eta$ (Sec. V B; needed by `paper_figures.py`) |
| `paper_figures.py` | Figures of the $O(2)$ fit terms, growth rate, amplitude ratio, level repulsion at $K=1$, $U(1)$ flows and growth factor, drift; Tables III and V (`paper_tables.tex`) |
| `figures.py ../data` | Figures of the eigenvalue phases ($U(1)$) and of the seed versus $\chi$ |
| `bootstrap.py` | Residual-bootstrap uncertainties (Sec. IV B) |
| `runs/additional/analysis/b7.py ../data` | Figure of the $C$-only splitting (Sec. V C) |
| `runs/additional/analysis/b8.py` | Figure of the BKT point $K=1/2$ (Sec. IV E) |
| `runs/additional/analysis/b9.py ../data` | Figure and Table VI of the compensation (Sec. VI) |
| `runs/additional/analysis/jj.py`, `wide_tube.py`, ... | Numbers quoted in the text (marginal injection, wider tube, branches, single-step replacements, singular values, ...) |
| `runs/additional/analysis/xxz_alpha_check.py` | Exact-diagonalization check of the amplitude-ratio coefficients (Sec. IV C) |

The two figures of Appendices D and E are based on earlier runs, included in `earlier/` with their own scripts
(`earlier/B_1_u1_tube_charge_checks`, Appendix D; `earlier/3_2_growth_rate`, Appendix E).  Run `python3 scripts/reproduce.py`
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

- `cho_ab_chi24/`: alternating $a$/$b$ runs at $K=1.5$, $\chi=24$ (K. Cho; Sec. VI)
- `yamamoto_conly_scan/`: $C$-only scan at $\chi=16$ over 17 values of $K$ and its fit code (Y. Yamamoto; Sec. V C)
- `fushio_trotter_injection/`: Trotter injection of the dimerization read with a wider tube (R. Fushio; Sec. VI)

The $C$-only runs with a transverse field (K. Cho; Sec. V C) are in `runs/additional/data/cho_transverse_field_20261024/` (`result.toml` and `config.toml` of each run) and are analyzed by `runs/additional/analysis/v01_signature.py`.

## Licenses

Code: MIT License (`LICENSE`).  Data (spectra and derived tables): Creative Commons Attribution 4.0 International
(`LICENSE-DATA`).  If you use the data, please cite the paper.

## Use of AI tools

Scripts in this repository were written with the help of AI coding assistants (Anthropic Claude Opus 5.5 via
Claude Code; OpenAI Codex with GPT-6 Astra, GPT-6 Sol, and GPT-6.1 Sol), as stated in the paper; all results were
checked by the authors.
