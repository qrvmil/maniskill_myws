#!/usr/bin/env bash
# Sequential real-buffer gates and the first fixed-budget source specialist.
set -euo pipefail
if [[ $# != 4 ]]; then
    echo "Usage: bash $0 ALIGNMENT_MANIFEST ZERO_REPORT OFFLINE_NPZ RUN_TAG" >&2
    exit 2
fi
PLD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PLD_ROOT"
PLD_ALIGN=$1
PLD_ZERO=$2
PLD_OFFLINE=$3
PLD_TAG=$4
[[ "$PLD_TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'RUN_TAG must be a simple directory suffix' >&2; exit 2; }
for PLD_INPUT in "$PLD_ALIGN" "$PLD_ZERO" "$PLD_OFFLINE"; do
    [[ -f "$PLD_INPUT" ]] || { echo "Missing input: $PLD_INPUT" >&2; exit 2; }
done
PLD_COMMON=(--alignment-manifest "$PLD_ALIGN" --zero-report "$PLD_ZERO")
PLD_SMOKE="outputs/pld_libero/EXP-002/rl-smoke-$PLD_TAG"

scripts/pld/libero_python.sh scripts/pld/run_libero.py train \
    --config configs/pld_libero/anchor_bowl_rl_smoke.json "${PLD_COMMON[@]}" \
    --offline-buffer "$PLD_OFFLINE" --output "$PLD_SMOKE"
# Engineering evaluation: its eight-step checkpoint must retain the smoke config.
scripts/pld/libero_python.sh scripts/pld/run_libero.py eval \
    --config configs/pld_libero/anchor_bowl_rl_smoke.json "${PLD_COMMON[@]}" \
    --checkpoint "$PLD_SMOKE/checkpoints/residual_step_8.pt" --distance D0 --episodes 2 \
    --output "outputs/pld_libero/EXP-000/learned-residual-smoke-$PLD_TAG"
scripts/pld/libero_python.sh scripts/pld/run_libero.py train \
    --config configs/pld_libero/anchor_bowl_otf_smoke.json "${PLD_COMMON[@]}" \
    --offline-buffer "$PLD_OFFLINE" --output "outputs/pld_libero/EXP-002/rl-otf-smoke-$PLD_TAG"
scripts/pld/libero_python.sh scripts/pld/run_libero.py train \
    --config configs/pld_libero/anchor_bowl_otf.json "${PLD_COMMON[@]}" \
    --offline-buffer "$PLD_OFFLINE" --output "outputs/pld_libero/EXP-002/source-otf-$PLD_TAG"
