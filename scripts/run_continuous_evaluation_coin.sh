#!/usr/bin/env bash
# CoIN x PSG-Nav runs for the paper's main table, qwen3.5-flash, the 100 frozen val_unseen
# episodes (configs/coin_val_unseen_subset100_seed77.json), official CoIN protocol (0.25 m,
# 500 steps). Every table cell is the mean of 3 runs; the full split is run by hand at the end.
#
#   PSG-Nav + AVS = E0 (scripts/run_coin_e0.sh): A0 + match_strict_no, base fixes off.
#                   Chosen on 2026-11-21: best SR (14.0, tied with E2), same code and prompt
#                   as the InstanceNav main configuration.
#   PSG-Nav       = E1: instance verification off, the base's category-count lock. The
#                   launcher refuses verification off together with A0, so E1 keeps the
#                   candidate approach on (same as E1 on InstanceNav).
# Both use the CoIN launcher defaults otherwise (full description in exploration).
#
# Each line launches one experiment on $GPUS through scripts/run_coin.sh. Results go to
# data/results_coin_<timestamp>/ (run_info.json records switches, git commit and source
# hashes), logs to outputs/coin_<timestamp>_gpu<N>.log. A timestamp that already has results
# is refused; to resume an interrupted run, rerun its line with --resume (an episode
# deferred after exhausted API retries is also rerun by --resume). Do not edit PSG_Nav.py,
# utils/, configs/ or graph/ between these runs. Each experiment runs even when an earlier
# one exits with an error. Lines run one after another; comment out what you do not want yet.
# One run takes about 4-5 h on two GPUs (E0_r1: 4 h 45 min).
#
# Usage:
#   export PSGNAV_PYTHON=/home/temp/.conda/envs/psgnav-instance/bin/python
#   bash scripts/run_continuous_evaluation_coin.sh
set -u
cd "$(dirname "$0")/.."
export PSGNAV_PYTHON="${PSGNAV_PYTHON:-/home/temp/.conda/envs/psgnav-instance/bin/python}"

VLM=qwen3.5-flash
# GPUs for every run below (default 0,1). On a one-GPU machine use --gpu 0 instead of --gpus.
GPUS=0,1
E1=(--enable-instance-confirmation false --enable-candidate-approach true)

# Main table, 3 runs per cell. E1 and E0 alternate so each pair runs back to back.
bash scripts/run_coin_e0.sh --vlm "$VLM" --gpus "$GPUS" "${E1[@]}" --timestamp coin_e1_r1
bash scripts/run_coin_e0.sh --vlm "$VLM" --gpus "$GPUS" --timestamp coin_e0_strict_r2
bash scripts/run_coin_e0.sh --vlm "$VLM" --gpus "$GPUS" "${E1[@]}" --timestamp coin_e1_r2
bash scripts/run_coin_e0.sh --vlm "$VLM" --gpus "$GPUS" --timestamp coin_e0_strict_r3
bash scripts/run_coin_e0.sh --vlm "$VLM" --gpus "$GPUS" "${E1[@]}" --timestamp coin_e1_r3

# Done (2026-11-20/21), kept for the record; one run each (SR / SPL):
#bash scripts/run_coin_e0.sh --vlm "$VLM" --gpus "$GPUS" --timestamp coin_e0_strict_r1   # 14.0 / 4.03
#bash scripts/run_coin_e2.sh --vlm "$VLM" --gpus "$GPUS" --timestamp coin_e2_strict_r1   # 14.0 / 4.36
#bash scripts/run_coin_e2.sh --vlm "$VLM" --gpus "$GPUS" --instance-verification-prompt match_context_optional --timestamp coin_e2_ctx_r1   # 12.0 / 5.56
#bash scripts/run_coin_e2.sh --vlm "$VLM" --gpus "$GPUS" --instance-verification-prompt match --timestamp coin_e2_match_r1   # 8.0 / 2.95
