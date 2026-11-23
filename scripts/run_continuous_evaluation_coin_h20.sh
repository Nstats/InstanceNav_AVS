#!/usr/bin/env bash

# H20 GPU 0：CoIN E1，每个 split 固定 100 个任务。如需并行，将两条命令分别复制到两个终端。
bash scripts/run_coin_e0.sh \
  --split val_seen \
  --gpu 0 \
  --nav-base psgnav \
  --vlm qwen3.5-flash \
  --enable-instance-confirmation false \
  --enable-candidate-approach true \
  --timestamp coin_e1_val_seen_r1

bash scripts/run_coin_e0.sh \
  --split val_seen_synonyms \
  --gpu 0 \
  --nav-base psgnav \
  --vlm qwen3.5-flash \
  --enable-instance-confirmation false \
  --enable-candidate-approach true \
  --timestamp coin_e1_val_seen_synonyms_r1

# 已完成的 E0 第二轮（SR / SPL）：val_seen 15.0 / 6.04，val_seen_synonyms 13.0 / 6.47。
# bash scripts/run_coin_e0.sh \
#   --split val_seen \
#   --gpu 0 \
#   --vlm qwen3.5-flash \
#   --timestamp coin_e0_val_seen_r2

# bash scripts/run_coin_e0.sh \
#   --split val_seen_synonyms \
#   --gpu 0 \
#   --vlm qwen3.5-flash \
#   --timestamp coin_e0_val_seen_synonyms_r2
