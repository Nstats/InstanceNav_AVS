#!/usr/bin/env bash
# CoIN x PSG-Nav runs, qwen3.5-flash, the 100 frozen val_unseen episodes
# (configs/coin_val_unseen_subset100_seed77.json), official CoIN protocol (0.25 m, 500 steps).
#   E0  PSG-Nav + AVS (scripts/run_coin_e0.sh): A0 + match_strict_no, base fixes off. E0_r1
#       (14.0 / 4.03) was the best of four configurations; r2 and r3 give an estimate free of
#       that selection and of single-run noise (main table: mean of 3).
#   E1  PSG-Nav: instance verification off (category-count lock), main table. The launcher
#       refuses verification off together with A0, so E1 keeps the candidate approach on
#       (same as E1 on InstanceNav).
#   B   E0 + active approach to unconfirmed candidates. Is A0's "no dedicated approach" also
#       right on CoIN? (InstanceNav: approach -4.9.)
#   E4  E0 + non-yes answers exclude the candidate for the episode (A0E4). The largest
#       negative ablation on InstanceNav (-15.5); checks it on the second benchmark.
# One run per configuration when its paired difference with E0 is significant; reruns
# (_r2, _r3) only where it is not. The full split is run by hand at the end.
#
# Each line launches one experiment through scripts/run_coin.sh. Results go to
# data/results_coin_<timestamp>/ (run_info.json records switches, git commit and source
# hashes), logs to outputs/coin_<timestamp>_gpu<N>.log. A timestamp that already has results
# is refused; to resume an interrupted run, rerun its line with --resume (an episode deferred
# after exhausted API retries is also rerun by --resume). Do not edit PSG_Nav.py, utils/,
# configs/ or graph/ between these runs. Each experiment runs even when an earlier one exits
# with an error. Lines run one after another. About 4-5 h per run on two GPUs.
#
# Usage:
#   export PSGNAV_PYTHON=/home/temp/.conda/envs/psgnav-instance/bin/python
#   bash scripts/run_continuous_evaluation_coin.sh
set -u
cd "$(dirname "$0")/.."
export PSGNAV_PYTHON="${PSGNAV_PYTHON:-/home/temp/.conda/envs/psgnav-instance/bin/python}"

VLM=qwen3.5-flash
E1=(--enable-instance-confirmation false --enable-candidate-approach true)

# ---- This machine, two GPUs (0,1) -------------------------------------------------------
# The third E0 run and the two ablations; they pair with E0_r1 and E0_r3 on this machine.
GPUS=0,1
bash scripts/run_coin_e0.sh --vlm "$VLM" --gpus "$GPUS" --timestamp coin_e0_strict_r3
bash scripts/run_coin_e0.sh --vlm "$VLM" --gpus "$GPUS" --enable-candidate-approach true --timestamp coin_b_r1
bash scripts/run_coin_e0.sh --vlm "$VLM" --gpus "$GPUS" --instance-rejection-policy non_yes --timestamp coin_a0e4_r1

# ---- Single-GPU machine (GPU 0); uncomment there, and comment out the block above --------
# PSG-Nav original (E1) and the second E0 run, back to back on the same machine. Copy
# data/results_coin_<timestamp>/ and outputs/coin_<timestamp>_gpu0.log back here afterwards.
# Set PSGNAV_PYTHON if that machine's interpreter is elsewhere.
#GPU=0
#bash scripts/run_coin_e0.sh --vlm "$VLM" --gpu "$GPU" "${E1[@]}" --timestamp coin_e1_r1
#bash scripts/run_coin_e0.sh --vlm "$VLM" --gpu "$GPU" --timestamp coin_e0_strict_r2

# Done (2026-11-20/21), one run each (SR / SPL):
#bash scripts/run_coin_e0.sh --vlm "$VLM" --gpus 0,1 --timestamp coin_e0_strict_r1   # 14.0 / 4.03
#bash scripts/run_coin_e2.sh --vlm "$VLM" --gpus 0,1 --timestamp coin_e2_strict_r1   # 14.0 / 4.36
#bash scripts/run_coin_e2.sh --vlm "$VLM" --gpus 0,1 --instance-verification-prompt match_context_optional --timestamp coin_e2_ctx_r1   # 12.0 / 5.56
#bash scripts/run_coin_e2.sh --vlm "$VLM" --gpus 0,1 --instance-verification-prompt match --timestamp coin_e2_match_r1   # 8.0 / 2.95
