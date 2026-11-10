#!/usr/bin/env bash
# Three sequential collection runs, each on the same frozen 100 CoIN episodes.
# No jobs are started merely by editing this script. Run it manually when ready.
set -euo pipefail
cd "$(dirname "$0")/.."
if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != --check-only ]]; }; then
  echo 'Usage: bash scripts/run_continuous_coin_collection.sh [--check-only]' >&2
  exit 2
fi
export INSTANCE_VERIFICATION_PROMPT=match_strict_no
unset ENABLE_CONTRADICTION_CHECK
options=(--split val_unseen --subset 100 --seed 77 --gpus 0,1
  --instance-collect-only true --enable-instance-confirmation true
  --instance-confirmation-threshold 1 --instance-confirmation-mode positive_match
  --verify-before-candidate false --enable-post-confirmation-contradiction-check false
  --preserve-positive-evidence-on-recovery false --rag-mode 0
  --enable-instance-description-for-exploration false --enable-object-relations false
  --enable-group-llm-filter true --enable-room-llm-filter true
  --landmark-comparison-repeats 1 --landmark-comparison-workers 10 --vlm qwen3.5-flash)
export ENABLE_RELATIVE_EVIDENCE=false
if [[ "${1:-}" == --check-only ]]; then
  exec bash scripts/run_coin.sh "${options[@]}" --check-only
fi
batch="$(date +%Y%m%d_%H%M%S)_$$"
for round in 1 2 3; do
  run_id="val_unseen_collect_${batch}_r${round}"
  echo "CoIN collection round ${round}/3: ${run_id} (100 episodes, 500 atomic actions each)"
  bash scripts/run_coin.sh "${options[@]}" --timestamp "$run_id"
done
