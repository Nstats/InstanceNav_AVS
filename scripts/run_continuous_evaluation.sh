#!/usr/bin/env bash
# Remaining InstanceNav x PSG-Nav runs for the main configuration A0 (paper_AVSNav.md, internal
# notes "最小待做实验", items 1, 5, 6, 7), qwen3.5-flash unless noted, fixed val10 100-task subset.
#   1  A0 run 3        Main configuration. A0 - B is not significant (+5.0 [-1.8, +13.4]), so A0
#                      gets a third run; it also fills task 65, which A0_r2 (ti2080) lacks.
#   5  A0E4            A0 + "no and unknown are excluded for the episode". Table 4 currently adds
#                      exclusion on top of B; this adds it on top of the main configuration.
#   6  A0E5            A0 + two yes votes before committing (k=2).
#   6  A0E3            A0 + "an explicit no is excluded for the episode". Lower priority than A0E5.
#   7  8B A0           A0 with Qwen3-VL-8B for every call, against 8b_B (SR 30) and ProCompNav.
# If an A0E* comparison with A0 is not significant, rerun it twice more (_r2, _r3).
#
# Each line launches one experiment on GPUs 0,1 through scripts/run_instance_ablation.sh,
# which archives source/config under outputs/instance_ablation_runs/<suite>/<timestamp>/
# and refuses a run when the algorithm core (PSG_Nav.py, utils/, configs/, graph/, the task
# subset, submodules) or the suite's common settings differ from the suite's. Do not edit
# those files between these runs. This file is excluded from the fingerprint. Each experiment
# runs even when an earlier one exits with an error. To resume an interrupted run, rerun its
# line with --resume. Lines run one after another; comment out what you do not want yet.
#
# Usage:
#   export PSGNAV_PYTHON=/home/temp/.conda/envs/psgnav-instance/bin/python
#   bash scripts/run_continuous_evaluation.sh
set -u
cd "$(dirname "$0")/.."

# Same suite as avs5_m7b_A0_r1 and avs5_m7b_RA_r1: same algorithm core and common settings,
# so A0_r3 and the A0E* runs pair with A0_r1 on identical code. The A0E* presets were added
# to tools/run_instance_ablation.py (not part of the core) on 2026-11-19.
SUITE=avs5_m7b_20261118
VLM=qwen3.5-flash
# GPUs for every run below (default 0,1). On a one-GPU machine set GPUS=0.
GPUS=0,1

bash scripts/run_instance_ablation.sh A0   --vlm "$VLM" --suite "$SUITE" --gpus "$GPUS" --timestamp avs5_m7b_A0_r3
bash scripts/run_instance_ablation.sh A0E4 --vlm "$VLM" --suite "$SUITE" --gpus "$GPUS" --timestamp avs5_m7b_A0E4_r1
bash scripts/run_instance_ablation.sh A0E5 --vlm "$VLM" --suite "$SUITE" --gpus "$GPUS" --timestamp avs5_m7b_A0E5_r1
bash scripts/run_instance_ablation.sh A0E3 --vlm "$VLM" --suite "$SUITE" --gpus "$GPUS" --timestamp avs5_m7b_A0E3_r1

# Reruns if A0E5 / A0E3 vs A0 is not significant (three runs each in total):
#bash scripts/run_instance_ablation.sh A0E5 --vlm "$VLM" --suite "$SUITE" --gpus "$GPUS" --timestamp avs5_m7b_A0E5_r2
#bash scripts/run_instance_ablation.sh A0E5 --vlm "$VLM" --suite "$SUITE" --gpus "$GPUS" --timestamp avs5_m7b_A0E5_r3
#bash scripts/run_instance_ablation.sh A0E3 --vlm "$VLM" --suite "$SUITE" --gpus "$GPUS" --timestamp avs5_m7b_A0E3_r2
#bash scripts/run_instance_ablation.sh A0E3 --vlm "$VLM" --suite "$SUITE" --gpus "$GPUS" --timestamp avs5_m7b_A0E3_r3

# 8B (SiliconFlow, needs SILICONFLOW_API_KEY in .env). The old 8B suite avs5_m3a_8b_20261114 was
# written by older code (different core and common settings), so 8B A0 needs a new suite. The
# optional 8b B line reruns B on the same code, so 8b A0 - 8b B is a same-source pair; without
# it, 8b A0 is compared with the old 8b_B (SR 30) across code versions.
SUITE_8B=avs5_m7b_8b_20261119
VLM_8B=qwen3-vl-8b-instruct
bash scripts/run_instance_ablation.sh A0 --vlm "$VLM_8B" --suite "$SUITE_8B" --gpus "$GPUS" --timestamp avs5_m7b_8b_A0_r1
#bash scripts/run_instance_ablation.sh B  --vlm "$VLM_8B" --suite "$SUITE_8B" --gpus "$GPUS" --timestamp avs5_m7b_8b_B_r1
