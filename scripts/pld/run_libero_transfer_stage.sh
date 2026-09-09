#!/usr/bin/env bash
# Paired fixed-checkpoint evaluation; no target SR is used for model selection.
set -euo pipefail
if [[ $# != 4 ]]; then
    echo "Usage: bash $0 ALIGNMENT_MANIFEST ZERO_REPORT RESIDUAL_CHECKPOINT RUN_TAG" >&2
    exit 2
fi
PLD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PLD_ROOT"
PLD_TAG=$4
[[ "$PLD_TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'RUN_TAG must be a simple directory suffix' >&2; exit 2; }
PLD_COMMON=(--config configs/pld_libero/anchor_bowl_otf.json --alignment-manifest "$1"
            --zero-report "$2" --checkpoint "$3")
PLD_RUNS=()
# Complete the two-pair engineering pass before any 50-pair final run.
for PLD_EPISODES in 2 50; do
    for PLD_DISTANCE_INDEX in 0 1 2 3 4 5; do
        PLD_EXPERIMENT_ID=$(printf 'EXP-%03d' "$((PLD_DISTANCE_INDEX+2))")
        PLD_OUTPUT="outputs/pld_libero/$PLD_EXPERIMENT_ID/paired-${PLD_TAG}-n${PLD_EPISODES}"
        scripts/pld/libero_python.sh scripts/pld/run_libero.py eval "${PLD_COMMON[@]}" \
            --distance "D$PLD_DISTANCE_INDEX" --episodes "$PLD_EPISODES" --output "$PLD_OUTPUT"
        if [[ "$PLD_EPISODES" == 50 ]]; then PLD_RUNS+=("$PLD_OUTPUT"); fi
    done
done
scripts/pld/libero_python.sh scripts/pld/summarize_libero.py --runs "${PLD_RUNS[@]}" \
    --output "outputs/pld_libero/EXP-007/transfer-summary-$PLD_TAG"
scripts/pld/libero_python.sh scripts/pld/plot_libero.py transfer --runs "${PLD_RUNS[@]}" \
    --output "outputs/pld_libero/EXP-007/transfer-figures-$PLD_TAG"
