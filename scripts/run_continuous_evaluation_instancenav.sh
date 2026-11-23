#!/usr/bin/env bash
# InstanceNav x VLFM base runs, qwen3.5-flash, fixed val10 100-task subset (1000 atomic steps).
#   VLFM+AVS  --nav-base vlfm, A0: VLFM finds the category; only a detection the VLM confirms
#             ("yes", within 4 m, match_strict_no) becomes its target. Same AVS settings as
#             PSG-Nav A0.
#   VLFM      --nav-base vlfm, E1: verification off, original VLFM.
# The paired difference of the two runs is how much AVS adds to a second base (separate
# table; the main table uses PSG-Nav). The CoIN pair runs on the H20
# (scripts/run_continuous_evaluation_coin_h20.sh).
#
# Each line launches one experiment through scripts/run_instance_ablation.sh, which archives
# source/config under outputs/instance_ablation_runs/<suite>/<timestamp>/ and refuses a run
# when the algorithm core (PSG_Nav.py, utils/, configs/, graph/, the task subset, submodules)
# or the suite's common settings differ from the suite's. The VLFM runs have their own suite.
# Do not edit those files, nav_bases/ or VLFM_Nav.py between the two runs. Each experiment
# runs even when an earlier one exits with an error. To resume an interrupted run, rerun its
# line with --resume. Lines run one after another; comment out what you do not want yet.
#
# Usage:
#   export PSGNAV_PYTHON=/home/temp/.conda/envs/psgnav-instance/bin/python
#   bash scripts/run_continuous_evaluation_instancenav.sh
set -u
cd "$(dirname "$0")/.."
export PSGNAV_PYTHON="${PSGNAV_PYTHON:-/home/temp/.conda/envs/psgnav-instance/bin/python}"

SUITE_VLFM=avs5_vlfm_20261123
VLM=qwen3.5-flash
# GPUs for every run below (default 0,1).
GPUS=0,1
# The four VLFM model servers (tmux session vlfm_servers, about 7 GB on GPU 1); does nothing
# when they already listen.
bash scripts/launch_vlfm_servers.sh --gpu 1
bash scripts/run_instance_ablation.sh A0 --vlm "$VLM" --nav-base vlfm --suite "$SUITE_VLFM" --gpus "$GPUS" --timestamp vlfm_A0_r1
bash scripts/run_instance_ablation.sh E1 --vlm "$VLM" --nav-base vlfm --suite "$SUITE_VLFM" --gpus "$GPUS" --timestamp vlfm_E1_r1
