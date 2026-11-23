#!/usr/bin/env bash

# InstanceNav × PSG-Nav：固定 val10 的 100 个任务，本机 GPU 0、1。
# 同一 suite 的实验使用同一版算法；未启用的实验保留为注释。
bash scripts/run_instance_ablation.sh \
  A0 \
  --vlm qwen3.5-flash \
  --suite avs5_m7b_20261118 \
  --gpus 0,1 \
  --timestamp avs5_m7b_A0_r3

bash scripts/run_instance_ablation.sh \
  A0E4 \
  --vlm qwen3.5-flash \
  --suite avs5_m7b_20261118 \
  --gpus 0,1 \
  --timestamp avs5_m7b_A0E4_r1

# 可选消融：两票确认（A0E5）、仅排除明确的 no（A0E3）。
# bash scripts/run_instance_ablation.sh \
#   A0E5 \
#   --vlm qwen3.5-flash \
#   --suite avs5_m7b_20261118 \
#   --gpus 0,1 \
#   --timestamp avs5_m7b_A0E5_r1

# bash scripts/run_instance_ablation.sh \
#   A0E3 \
#   --vlm qwen3.5-flash \
#   --suite avs5_m7b_20261118 \
#   --gpus 0,1 \
#   --timestamp avs5_m7b_A0E3_r1

# 差异不显著时，再做以下重复实验。
# bash scripts/run_instance_ablation.sh \
#   A0E5 \
#   --vlm qwen3.5-flash \
#   --suite avs5_m7b_20261118 \
#   --gpus 0,1 \
#   --timestamp avs5_m7b_A0E5_r2

# bash scripts/run_instance_ablation.sh \
#   A0E5 \
#   --vlm qwen3.5-flash \
#   --suite avs5_m7b_20261118 \
#   --gpus 0,1 \
#   --timestamp avs5_m7b_A0E5_r3

# bash scripts/run_instance_ablation.sh \
#   A0E3 \
#   --vlm qwen3.5-flash \
#   --suite avs5_m7b_20261118 \
#   --gpus 0,1 \
#   --timestamp avs5_m7b_A0E3_r2

# bash scripts/run_instance_ablation.sh \
#   A0E3 \
#   --vlm qwen3.5-flash \
#   --suite avs5_m7b_20261118 \
#   --gpus 0,1 \
#   --timestamp avs5_m7b_A0E3_r3

# 8B 使用自己的 suite；可选 B 用于同版本对照。
bash scripts/run_instance_ablation.sh \
  A0 \
  --vlm qwen3-vl-8b-instruct \
  --suite avs5_m7b_8b_20261119 \
  --gpus 0,1 \
  --timestamp avs5_m7b_8b_A0_r1

# bash scripts/run_instance_ablation.sh \
#   B \
#   --vlm qwen3-vl-8b-instruct \
#   --suite avs5_m7b_8b_20261119 \
#   --gpus 0,1 \
#   --timestamp avs5_m7b_8b_B_r1
