#!/usr/bin/env bash

# val_seen_synonyms 固定 100 个任务；本机依次跑两轮，除核验提示词外配置相同。
# match_strict_no：第 3 轮（前两轮为本机 r1、H20 r2）。

# match_visible_evidence：本机第 2 轮。
bash scripts/run_coin_e0.sh \
  --split val_seen_synonyms \
  --subset 100 \
  --gpus 0,1 \
  --nav-base psgnav \
  --vlm qwen3.5-flash \
  --instance-verification-prompt match_visible_evidence \
  --timestamp coin_syn100_visible_evidence_local_r2
