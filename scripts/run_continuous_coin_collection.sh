#!/usr/bin/env bash
# CoIN 仅采集，不执行 STOP。三轮使用同一批 val_unseen 任务，一轮失败即停止。
set -e
export ENABLE_RELATIVE_EVIDENCE=false
unset ENABLE_CONTRADICTION_CHECK

collect() {
  bash scripts/run_coin.sh \
    --split val_unseen \
    --subset 100 \
    --seed 77 \
    --gpus 0,1 \
    --instance-collect-only true \
    --enable-instance-confirmation true \
    --instance-confirmation-threshold 1 \
    --instance-confirmation-mode positive_match \
    --instance-verification-prompt match \
    --verify-before-candidate false \
    --enable-post-confirmation-contradiction-check false \
    --preserve-positive-evidence-on-recovery false \
    --rag-mode 0 \
    --enable-instance-description-for-exploration true \
    --enable-object-relations false \
    --enable-group-llm-filter true \
    --enable-room-llm-filter true \
    --landmark-comparison-repeats 1 \
    --landmark-comparison-workers 10 \
    --vlm qwen3.5-flash \
    "$@"
}

if [[ $# == 1 && $1 == --check-only ]]; then
  collect --check-only
  exit
elif (( $# )); then
  printf 'Usage: bash scripts/run_continuous_coin_collection.sh [--check-only]\n' >&2
  exit 2
fi

batch="$(date +%Y%m%d_%H%M%S)_$$"
collect --timestamp "val_unseen_collect_${batch}_r1"
collect --timestamp "val_unseen_collect_${batch}_r2"
collect --timestamp "val_unseen_collect_${batch}_r3"
