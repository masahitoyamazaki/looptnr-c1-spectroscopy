#!/bin/bash
# Usage: worker.sh MACHINE OUTDIR   -- runs the tasks.tsv rows of MACHINE whose node equals this rank
MACHINE=$1; OUT=$2; RANK=${PMI_RANK:-0}
cd "$(dirname "$0")"; mkdir -p "$OUT" logs
echo "$MACHINE rank $RANK on $(hostname) at $(date)"
while IFS=$'\t' read -r m node nt tag sym K chi ns env; do
  [ "$m" = "$MACHINE" ] && [ "$node" = "$RANK" ] || continue
  ( export OPENBLAS_NUM_THREADS=$nt
    for kv in $env; do eval export "$kv"; done
    exec ~/.juliaup/bin/julia --project=$HOME/tnr_cpu --startup-file=no -t 1 run2.jl $sym $K $chi $ns "$OUT" $tag ) \
    > logs/$tag.log 2>&1 &
done < <(grep -v '^#' tasks.tsv)
wait
echo "$MACHINE rank $RANK finished at $(date)"
