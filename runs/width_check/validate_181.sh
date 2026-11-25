#!/bin/bash
# Validation of the restricted [1,8,1] readout (width_read_181.jl: vacuum and charge-one sectors only) against the full
# TNRKit CFTData(TA, TB; shape = [1,8,1], trunc = truncrank(16)), on the chi = 16 checkpoint at K = 1, step 24 of the
# main runs (main).  First run 2026-11-12 (log kept on the workstation); rerun 2026-11-25 with this
# script for the record (Otsuki R-26).  Output: out/validate_181.txt (the two lowest x of sector (1;2), both readouts,
# and their relative difference).
cd "$(dirname "$0")"; mkdir -p out
B=$HOME/runs/main
DIR=$B/out/CU1_K1.0000_chi16; N=24
{
  echo "# validate_181.jl  checkpoint: main/out/CU1_K1.0000_chi16/step_$(printf %03d $N).jls  (K = 1, chi = 16)"
  echo "# full: TNRKit CFTData(shape = [1,8,1], trunc = truncrank(16)); restricted: read181(TA, TB, 16, 3) of width_read_181.jl"
  echo "# $(date -Iseconds)  julia $(~/.juliaup/bin/julia --version | cut -d' ' -f3)  env ~/tnr_cpu (TNRKit 0.7.0)"
  DIR=$DIR N=$N ~/.juliaup/bin/julia --project=$HOME/tnr_cpu -t 6 validate_181.jl 2>&1
} > out/validate_181.txt
