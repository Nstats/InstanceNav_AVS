#!/usr/bin/env bash
# M7 of the experiment queue in outputs/paper_AVSNav.md (internal notes, 阶段五): component
# ablations that answer two questions, with qwen3.5-flash on the fixed val10 100-task subset.
#   Q1 (Table A) How far should a negative answer reach?
#     T0: "no" or "unknown" releases the candidate for the current approach only
#         (INSTANCE_REJECTION_SCOPE=approach, INSTANCE_REJECTION_POLICY=non_yes). It comes back
#         once it has been out of view for >= 5 policy steps and the agent has moved >= 1 m.
#         Compare with B (nothing excluded) and E4 (excluded for the episode).
#   Q2 (Table C) Is approaching before verifying necessary?
#     A0: candidates are never approached (ENABLE_CANDIDATE_APPROACH=false); verification
#         still runs whenever a candidate is within 4 m, and a yes still locks.
#     RA: approach only candidates the VLM already admitted (VERIFY_BEFORE_CANDIDATE=true).
# Everything else is the frozen main configuration (k=1, match_strict_no).
#
# This source adds approach logging (approach_start / approach_switch / approach_end /
# approach_release / approach_release_end in the [Instance Verification] records). Logging
# does not change decisions, but the new switches change the source fingerprint, so these
# runs need a new suite. The optional B line below reruns B in the same suite, which gives
# B approach logs for Table D and a same-source baseline.
#
# The M3a 8B queue (suite avs5_m3a_8b_20261114) is finished; it is no longer launched here.
#
# Each line launches one experiment on GPUs 0,1 through scripts/run_instance_ablation.sh,
# which archives source/config under outputs/instance_ablation_runs/<suite>/<timestamp>/
# and refuses to mix runs whose source fingerprint or common settings differ from the
# suite's. Do not edit tracked source between these runs. This file is excluded from the
# fingerprint, so editing it does not change the suite. Each experiment runs even when an
# earlier one exits with an error. To resume an interrupted run, rerun its line with --resume.
#
# Usage:
#   export PSGNAV_PYTHON=/home/temp/.conda/envs/psgnav-instance/bin/python
#   bash scripts/run_continuous_evaluation.sh
set -u
cd "$(dirname "$0")/.."

SUITE=avs5_m7_20261117
VLM=qwen3.5-flash

bash scripts/run_instance_ablation.sh T0 --vlm "$VLM" --suite "$SUITE" --gpus 0,1 --timestamp avs5_T0_r1
bash scripts/run_instance_ablation.sh A0 --vlm "$VLM" --suite "$SUITE" --gpus 0,1 --timestamp avs5_A0_r1
bash scripts/run_instance_ablation.sh RA --vlm "$VLM" --suite "$SUITE" --gpus 0,1 --timestamp avs5_RA_r1

#bash scripts/run_instance_ablation.sh T0 --vlm "$VLM" --suite "$SUITE" --gpus 0,1 --timestamp avs5_T0_r2
#bash scripts/run_instance_ablation.sh A0 --vlm "$VLM" --suite "$SUITE" --gpus 0,1 --timestamp avs5_A0_r2
#bash scripts/run_instance_ablation.sh RA --vlm "$VLM" --suite "$SUITE" --gpus 0,1 --timestamp avs5_RA_r2

# Optional, same suite:
# bash scripts/run_instance_ablation.sh B    --vlm "$VLM" --suite "$SUITE" --gpus 0,1 --timestamp avs5_B_m7_r1
# bash scripts/run_instance_ablation.sh T0no --vlm "$VLM" --suite "$SUITE" --gpus 0,1 --timestamp avs5_T0no_r1
