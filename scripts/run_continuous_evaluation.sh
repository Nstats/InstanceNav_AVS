#!/usr/bin/env bash
# AVS on CoIN-Bench, three consecutive runs of the fixed 100-episode val_unseen subset
# (configs/coin_val_unseen_subset100_seed77.json: seed 77, all 47 targets covered).
# Every round is the same configuration: run_coin.sh defaults (Context-Nav protocol, CoIN
# vocabulary, 0.25 m candidate merge radius, GPUs 0,1), including environment overrides.
# Each round gets its own result id (val_unseen_<date>_<pid>) and results directory
# data/results_coin_<id>/ with summary.json. API response caches are in memory, so the
# rounds do not reuse each other's model answers.
# Each round runs even when an earlier round exits with an error.
set -u
cd "$(dirname "$0")/.."

for round in 1 2 3; do
  echo "=== CoIN val_unseen subset100 round ${round}/3 ($(date '+%F %T')) ==="
  bash scripts/run_coin.sh --split val_unseen --subset 100
  echo "=== round ${round}/3 exit $? ($(date '+%F %T')) ==="
done
