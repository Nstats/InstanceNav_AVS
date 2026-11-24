#!/usr/bin/env bash

# val_seen：已完成 92/100，续跑剩余 8 个任务。
bash scripts/run_coin_e0.sh \
  --split val_seen \
  --subset 100 \
  --gpus 0,1 \
  --nav-base psgnav \
  --vlm qwen3.5-flash \
  --instance-verification-prompt match_visible_evidence \
  --timestamp coin_seen100_visible_evidence_local_r1 \
  --resume

# val_seen_synonyms：已完成 100/100，不再运行。
# bash scripts/run_coin_e0.sh \
#   --split val_seen_synonyms \
#   --subset 100 \
#   --gpus 0,1 \
#   --nav-base psgnav \
#   --vlm qwen3.5-flash \
#   --instance-verification-prompt match_visible_evidence \
#   --timestamp coin_syn100_visible_evidence_local_r1

# val_unseen：已完成 43/100，续跑剩余 57 个任务。
bash scripts/run_coin_e0.sh \
  --split val_unseen \
  --subset 100 \
  --gpus 0,1 \
  --nav-base psgnav \
  --vlm qwen3.5-flash \
  --instance-verification-prompt match_visible_evidence \
  --timestamp coin_unseen100_visible_evidence_local_r1 \
  --resume
