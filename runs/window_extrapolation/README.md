# window_extrapolation_20261110: fit-window dependence (Task 1) and the K = 1/2 rate (Task 2)

Analysis of existing data only (no new RG runs). Existing files were not modified. Python with numpy/scipy/matplotlib.

## Scripts

| File | What it does |
|---|---|
| `common.py` | Finds the run folders (main runs `main/data/CU1_*`, B5 χ=96, B8 K=0.5/0.6). Imports `load`, `level`, `fit_growth`, `ALPHA` unchanged from `main/analysis/analyze.py`. `fit_cut(K, X, XJ, X_max)` is the paper fit (same as `analyze_o2` / `paper_figures.o2_fit`) with a variable upper cut. |
| `task1_window.py` | Task 1 → `task1_cuts.csv`, `task1_extrap.csv`, `task1_summary.csv`, `fig_window_extrap.pdf/png`, `task1_output.txt` |
| `task2_bkt.py` | Task 2 → `task2_bkt.csv`, `task2_ghat_steps.csv`, `task2_output.txt` |

Run: `python3 task1_window.py; python3 task2_bkt.py` from this folder.

## Task 1: protocol

- Fit of Sec. IV B: start n_s = 12; end n_e = the step before |X| first exceeds X_max; columns 1, t, r^(n−n_e) [+ (2^(2−4K))^(n−12) for K < 3/4]. r is chosen on the same 4000-point grid in ln r over [1.02, 4]. A_J comes from linear least squares with the same columns. The amplitude ratio is normalized by α_J/(4Kα_{0,1}).
- X_max ∈ {0.1, 0.05, 0.03, 0.02, 0.01, 0.005}. A cut is skipped when it leaves fewer than 5 points (6 when the umklapp term is included). In practice this skips only K=0.6, χ=24, X_max=0.005. The number of points is in `task1_cuts.csv` (`npts`).
- Check: at X_max = 0.1 the fits reproduce `o2_fits.csv` and Table tab:o2. At X_max = 0.02 they reproduce the manuscript values for χ = 40, 64.
- Runs: K = 0.75–1.75 × χ = 24–64 (main runs), χ = 96 (B5, K = 1, 1.5), and K = 0.6 × χ = 24–64 (B8). For K = 1.75, χ = 64, |X| stays below 0.1 up to n = 80, so only the cuts 0.01 and 0.005 exist and there is no extrapolation.
- Extrapolation: an unweighted linear least-squares fit y = y0 + b·X_max over all usable cuts (`*_extrap`, primary). The uncertainty is the OLS standard error of y0 from the residual scatter. Two stability checks:
  - the same fit over 0.01 ≤ X_max ≤ 0.1 (`*_extrap_ge001`);
  - the same fit over X_max ≤ 0.03 (`*_extrap_smallcuts`).

  The cut windows are nested, so the points are not independent and the SE is only indicative. The `*_flags` columns mark a run when any of these hold:
  - fewer than 3 cuts;
  - SE > 1%;
  - the all-cut and small-cut extrapolations differ by more than max(2 SE, 0.005);
  - the spread over cuts is larger than 0.03;
  - r sits at the edge of the grid.
- Figure: χ ≥ 32 only, because χ = 24 is unstable at small cuts (its values are in the CSV). Open symbols are the individual cuts. Filled symbols with error bars, offset to X_max < 0, are the extrapolations.

## Task 1: main numbers (median [min, max] over χ = 32–96; from `task1_summary.csv`, rows `chi32-96`)

| K | r/2^{2−K}: X≤0.1 | X≤0.02 | extrap. | amp. ratio: X≤0.1 | X≤0.02 | extrap. |
|---|---|---|---|---|---|---|
| 0.60 | 0.968 [0.960, 0.974] | 0.985 [0.983, 0.989] | 0.988 [0.955, 1.038] | 0.995 [0.985, 0.997] | 0.997 [0.995, 1.011] | 1.003 [0.998, 1.007] |
| 0.75 | 0.981 [0.976, 0.987] | 0.999 [0.989, 1.007] | 1.008 [1.003, 1.013] | 1.005 [1.004, 1.006] | 0.998 [0.994, 1.003] | 0.992 [0.988, 1.002] |
| 1.00 | 0.993 [0.992, 0.996] | 0.999 [0.997, 1.009] | 0.999 [0.985, 1.033] | 1.005 [1.003, 1.006] | 1.001 [0.997, 1.009] | 0.998 [0.948, 1.022] |
| 1.25 | 1.002 [1.002, 1.003] | 0.999 [0.988, 1.005] | 0.999 [0.961, 1.006] | 0.998 [0.998, 0.999] | 0.994 [0.975, 1.007] | 0.997 [0.916, 1.018] |
| 1.50 | 1.014 [1.012, 1.015] | 1.003 [1.002, 1.008] | 1.002 [0.986, 1.003] | 0.999 [0.998, 1.008] | 1.001 [0.998, 1.059] | 1.007 [1.001, 1.032] |
| 1.75* | 1.027 [1.027, 1.028] | 1.008 [1.006, 1.012] | 1.005 [1.003, 1.012] | 1.059 [1.056, 1.061] | 1.029 [0.998, 1.071] | 1.075 [1.045, 1.124] |

\*K = 1.75: χ = 32–48 only (χ = 64 never reaches |X| = 0.1).

For the runs with the most points (χ = 64, 96), the extrapolated values are:

| Run | r/2^{2−K} | amplitude ratio |
|---|---|---|
| K=1, χ=64 | 0.9989 ± 0.0006 | 1.0006 ± 0.0042 |
| K=1, χ=96 | 0.9994 ± 0.0009 | 0.9980 ± 0.0013 |
| K=1.25, χ=64 | 0.9984 ± 0.0017 | 0.9995 ± 0.0019 |
| K=1.5, χ=64 | 1.0018 ± 0.0004 | 1.0012 ± 0.0005 |
| K=1.5, χ=96 | 0.9986 ± 0.0004 | 1.0038 ± 0.0021 |
| K=0.75, χ=64 | 1.0069 ± 0.0043 | 0.9936 ± 0.0043 |

### Observations

- The rate ratio moves systematically with the cut and the extrapolation is stable:
  - For K ≥ 1.5 the excess at X ≤ 0.1 (+1.4%, +2.7%) shrinks roughly linearly. The extrapolated values are 1.002 (K=1.5) and 1.005 (K=1.75).
  - For K ≤ 1 the deficit (−2% at K=0.75, −0.7% at K=1) disappears: the extrapolated values are 1.008 and 0.999.
  - At K = 0.6 the deficit shrinks from −3.2% to about −1.2% (0.988). This extrapolation is less stable: 3 of 4 runs are flagged, and only 11–15 points remain at small cuts.
- The amplitude ratio for 0.6 ≤ K ≤ 1.5 shows no significant trend with the cut. The standard-cut values are already within 0.5% of 1. The extrapolations stay within about 1% of 1 for χ ≥ 48, but scatter by several percent at χ = 32–40 (e.g. K=1.25 χ=32: 0.92 ± 0.05). The ratio becomes noisy at small cuts because A_J is then small compared with the irregularity of X_J at the floor.
- K = 1.75 amplitude ratio:
  - It does not converge to 1. The extrapolations are 1.045–1.124 (all flagged as SE > 1%).
  - Over 0.01 ≤ X_max ≤ 0.1 they are 1.007–1.073 (χ=32: 1.007, χ=40: 1.073, χ=48: 1.022).
  - At χ = 64 (cuts 0.01 and 0.005 only) the values are 1.035 and 1.048.
  - This matches the manuscript statement that about 3% remains at small cuts. The extrapolation does not resolve it.
- χ = 24 is unstable at cuts ≤ 0.02 for every K: the rate sits at the grid edge or swings by tens of percent. Its extrapolations are meaningless.

## Task 2: protocol

- Measured r: the paper fit at K = 1/2 with the standard cut |X| ≤ 0.1 (window 12–n_e). The umklapp column (2^{2−4K})^{n−12} is a constant at K = 1/2. Smaller cuts are impossible at K = 1/2: X(12) = ĝ/2 ≈ 0.074, so |X| ≤ 0.05 leaves no window.
- ĝ(n) comes from the triplet, x_t = 1/2 − ĝ/4, with x_t the mean of the lowest (1,2)-sector level and the lowest (0,1)-sector level (as in `followup_B/analysis/b8.py`). Late in the window the triplet is also shifted by g_d² (ĝ_t,direct turns negative by n_e), so ĝ is taken from fits to the clean range n = 8–20 (8–14 for χ = 24) and extended into the window. There are two estimates:
  - `flow` (primary): the two-loop flow dĝ/dℓ = −ĝ² − ĝ³/2 with ℓ = n ln√2 + const and one fitted constant. The rms residual on n = 8–20 is (2–4)×10⁻⁴.
  - `lin`: a straight line in 1/ĝ_t vs n. The slope of 4/ĝ is 1.48–1.50 per step for χ ≥ 32.

  The two estimates agree within 0.001 in ĝ. The singlet value ĝ_s = (4/3)(x_s − 1/2) is also written to `task2_ghat_steps.csv` as a cross-check.
- Log-corrected expectation: the plain average over n = 12..n_e of 2^{3/2 − 3ĝ(n)/4}. Also recorded: the min/max, the average over the last 5 steps, and the value obtained using the directly measured (biased) ĝ.

## Task 2: main numbers (`task2_bkt.csv`)

| χ | window | r | r/2^{3/2} | ĝ_flow range in window (mean) | ⟨2^{3/2−3ĝ/4}⟩ | r/log-corrected |
|---|---|---|---|---|---|---|
| 32 | 12–28 | 2.842 | 1.005 | 0.149 → 0.080 (0.107) | 2.675 (= 0.946·2^{3/2}) | 1.062 |
| 40 | 12–26 | 2.810 | 0.994 | 0.149 → 0.085 (0.111) | 2.670 (0.944·2^{3/2}) | 1.052 |
| 48 | 12–29 | 2.896 | 1.024 | 0.149 → 0.077 (0.106) | 2.677 (0.947·2^{3/2}) | 1.082 |
| 64 | 12–26 | 3.163 | 1.118 | 0.149 → 0.084 (0.111) | 2.670 (0.944·2^{3/2}) | 1.185 |

χ = 24 (window 12–19) gives r/2^{3/2} = 0.930 and r/log-corrected = 0.992.

### Background-model check

The paper's linear background 1 + t does not describe the logarithmic drift X ≈ ĝ(n)/2 over these windows: the fit residual rms is about 1×10⁻³. Two alternative backgrounds were fitted over the same window:

- `ghatbg`: subtract ĝ_flow(n)/2 from X, which is the theory value from eq:bkt with no free parameter, then fit with 1 + t.
- `freeghatbg`: include ĝ_flow(n) as a free column. Its fitted coefficient is 0.52–0.56 for χ ≥ 32, close to the theoretical 1/2.

With either background the residual rms drops to (0.3–1.5)×10⁻⁴, and the rates change substantially:

| χ | r/2^{3/2}: paper fit | r/2^{3/2}: ĝ background | r/log-corr.: paper fit | r/log-corr.: ĝ background | ρ/2^{3/2}: ĝ background + log-corrected column |
|---|---|---|---|---|---|
| 32 | 1.005 | 0.906 | 1.062 | 0.958 | 0.945 |
| 40 | 0.994 | 0.905 | 1.052 | 0.959 | 0.946 |
| 48 | 1.024 | 0.910 | 1.082 | 0.961 | 0.948 |
| 64 | 1.118 | 0.930 | 1.185 | 0.985 | 0.971 |

### Reading the Task 2 results (stated with caution)

- With the paper's background, the closeness of r to 2^{3/2} at χ = 32–48 is not evidence that log corrections are absent. Including them (≈ −5.5% in the expected factor) makes the agreement worse (+5–8%). The fit is also visibly biased by the mismatched background.
- With the theory background ĝ/2, the χ = 64 outlier (1.12) disappears and all χ give consistent values. The measured rate is then 1.5–4% below the log-corrected expectation (0.958–0.985), and 7–9.5% below 2^{3/2}.
- The remaining few-percent deficit could come from O(ĝ²) terms, the ĝ-dependence of the coefficients α at K = 1/2, or errors in extrapolating ĝ into the window. None of these was tested.
