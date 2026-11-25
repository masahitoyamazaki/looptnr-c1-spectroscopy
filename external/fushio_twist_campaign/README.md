# docs16: signed-twist calibration and fixed compensation


## Git publication snapshot (2026-11-10 JST)

This publication contains notes, code, figures, and aggregate results. The docs16 campaign is still running; `progress.md` and `status.json` describe the recorded snapshot, not live status or physical convergence. Raw per-measurement TOML, tensors, full-level CSV, logs, private process records, and frozen source/environment copies remain local. Links to those raw inputs refer to a local data installation. New docs16 tensors have not yet been archived to Drive.

Source SHA-256 values describe the original calculation files. Workstation prefixes in published metadata are made repository-relative; hashes of the publication copies are listed separately in `publication.json`. Existing reservations and predictions must not be replayed as a new campaign.

The live worker used byte-identical frozen copies of `Yamazaki/runs/followup_B_20261106/common.jl` and `support/isotropic_twist_20261109/twist_measure.jl`; the publication imports those canonical files. Its environment Manifest was byte-identical to `Yamazaki/runs/cpu_baseline_20261106/environment/Manifest.toml`. Restore the recorded Project/Manifest and original runtime configuration before reproducing archived runs. `TNR_COORDINATION_ROOT` selects a dedicated clean coordination checkout; `JULIA` selects the Juliaup executable and Python uses the current interpreter. The guard and source-hash checks remain required.
