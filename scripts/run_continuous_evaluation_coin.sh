#!/usr/bin/env bash
# CoIN x PSG-Nav runs, qwen3.5-flash, the 100 frozen val_unseen episodes
# (configs/coin_val_unseen_subset100_seed77.json), official CoIN protocol (0.25 m, 500 steps).
#   E0  A0 baseline (scripts/run_coin_e0.sh): unconfirmed candidates never become the goal,
#       one "yes" within 4 m locks, goal point frozen at the lock, depth gate off.
#   E2  E0 + base fixes (scripts/run_coin_e2.sh): goal point refreshed after the lock
#       (nearest goal-category box within 1.5 m, no model call) and depth valid-fraction
#       gate 0.5 with median depth (mirrors and glass get no distance).
#   Prompts: match_strict_no (preset default), match_context_optional, match. The prompt
#       that does best on CoIN becomes the CoIN prompt.
# Order: E0 and E2 with the same prompt first (base fixes), then the other prompts on E2.
# If a comparison is not significant, rerun both sides (_r2, _r3).
#
# Each line launches one experiment on $GPUS through scripts/run_coin.sh. Results go to
# data/results_coin_<timestamp>/ (run_info.json records switches, git commit and source
# hashes), logs to outputs/coin_<timestamp>_gpu<N>.log. A timestamp that already has results
# is refused; to resume an interrupted run, rerun its line with --resume. Do not edit
# PSG_Nav.py, utils/, configs/ or graph/ between paired runs. Each experiment runs even when
# an earlier one exits with an error. Lines run one after another; comment out what you do
# not want yet. Past CoIN 100-task runs took about 3-5 h each on two GPUs.
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

# Base fixes: E0 vs E2, same prompt.
bash scripts/run_coin_e0.sh --vlm "$VLM" --gpus "$GPUS" --timestamp coin_e0_strict_r1
bash scripts/run_coin_e2.sh --vlm "$VLM" --gpus "$GPUS" --timestamp coin_e2_strict_r1

# Prompts on E2.
bash scripts/run_coin_e2.sh --vlm "$VLM" --gpus "$GPUS" --instance-verification-prompt match_context_optional --timestamp coin_e2_ctx_r1
bash scripts/run_coin_e2.sh --vlm "$VLM" --gpus "$GPUS" --instance-verification-prompt match --timestamp coin_e2_match_r1

# Reruns if E2 - E0 or a prompt comparison is not significant (three runs each in total):
#bash scripts/run_coin_e0.sh --vlm "$VLM" --gpus "$GPUS" --timestamp coin_e0_strict_r2
#bash scripts/run_coin_e2.sh --vlm "$VLM" --gpus "$GPUS" --timestamp coin_e2_strict_r2
#bash scripts/run_coin_e2.sh --vlm "$VLM" --gpus "$GPUS" --instance-verification-prompt match_context_optional --timestamp coin_e2_ctx_r2
#bash scripts/run_coin_e2.sh --vlm "$VLM" --gpus "$GPUS" --instance-verification-prompt match --timestamp coin_e2_match_r2
