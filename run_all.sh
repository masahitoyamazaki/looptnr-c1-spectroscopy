#!/bin/sh
# Reproduce the figures, tables, and quoted numbers of the paper from the stored spectra (see README.md).
set -e
PY=${PYTHON:-python3}
export MPLBACKEND=Agg
cd "$(dirname "$0")/runs/main/analysis"
$PY analyze.py ../data
$PY drift.py ../data
$PY drift_table.py
$PY induced_drift.py
$PY synthetic_flow.py
$PY paper_figures.py
$PY figures.py ../data
$PY checks.py
$PY bootstrap.py
cd ../../additional/analysis
$PY b7.py ../data
$PY b8.py
$PY b9.py ../data
for s in jj.py wide_tube.py cho_ab_chi24.py b7_joint_fit.py b7_v01_ratio.py v01_signature.py; do $PY $s; done
for s in b1_b2.py b3_b4.py c7.py b9_validation.py; do $PY $s ../data; done
$PY a2_sv.py
echo "done: figures and tables are in runs/main/analysis and runs/additional/analysis"
