# Mid-RG corrections at the BKT point: interval 4 versus interval 2 (K. Cho)

Separately scripted runs by K. Cho (Paper II, Sec. VI): BKT point K = 1/2 (vertex weight c = 2), chi = 24, continued from
step 12 to step 64 with the projected-gradient correction of the twisted splitting applied every four (`int4`) or every
two (`int2`) steps, for the constraint sets P3_3c, P3_4c (the four constraints of Paper II), and P3_5c.

- `run_phase3_deep_bkt_and_p3_5c.jl`: driver script (TNR repository of the authors, commit 2d521d2b); it needs the helper
  files of `runs/midrg_compensation/cho_src/` and is included for documentation.
- `phase3_deep_bkt_and_p3_5c.toml`: results per corrected step (twisted splitting before and after the correction,
  K_n, triplet levels, anisotropy, central charge).
- `phase3_deep_bkt_and_p3_5c.log`: run log with the same numbers.

With P3_4c every four steps, the twisted splitting before each correction grows from 5e-3 at step 12 to 7e-2 at step 64
and the triplet splits (x_{0,1} = 0.406, x_t2 = 0.526 at step 64); every two steps it stays below 5e-4 from step 16 on.
