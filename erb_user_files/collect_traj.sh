#!/bin/bash
# Gather this iteration's exploration trajectories and model-deviation files
# into ./collected/ and print a one-line divergence summary per trajectory.
#
# Run from inside the exploration iteration folder (e.g. 001-exploration/)
# once the explore Slurm jobs have finished. ArcaNN's exploration/prepare.py
# copies this script here verbatim, so the iteration number is taken from the
# folder name rather than hard-coded.
set -euo pipefail

iter=$(basename "$PWD")
iter=${iter%-exploration}

dest=collected
mkdir -p "$dest"

printf '%-14s %10s %14s %14s  %s\n' traj devi_rows max_devi_f mean_devi_f status

shopt -s nullglob
for devi in he/*/*/model_devi_he_*.out; do
    dir=$(dirname "$devi")
    nnp=$(basename "$(dirname "$dir")")
    traj=$(basename "$dir")
    tag="${nnp}_${traj}"
    base="he_${nnp}_${iter}"

    for f in "$base.dcd" "$base.log" "model_devi_$base.out"; do
        [ -f "$dir/$f" ] && cp -p "$dir/$f" "$dest/${tag}_$f"
    done

    read -r rows maxf meanf < <(
        awk '!/^#/ && NF >= 5 {
                n++
                if (n == 1 || $5 > mx) mx = $5
                s += $5
             }
             END { printf "%d %.6g %.6g\n", n, (n ? mx : 0), (n ? s / n : 0) }' "$devi"
    )

    if [ -f "$dir/$base.dcd" ]; then status=done; else status=NO-DCD; fi
    printf '%-14s %10s %14s %14s  %s\n' "$tag" "$rows" "$maxf" "$meanf" "$status"
done

echo
echo "artifacts copied to $PWD/$dest/"
