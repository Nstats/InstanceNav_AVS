#!/usr/bin/env bash

# 本机 InstanceNav：先在 GPU 1 启动 VLFM 服务；启动失败就不跑测评。
bash scripts/launch_vlfm_servers.sh \
  --gpu 1 || exit $?

# VLFM + AVS
bash scripts/run_instance_ablation.sh \
  A0 \
  --vlm qwen3.5-flash \
  --nav-base vlfm \
  --suite avs5_vlfm_20261123 \
  --gpus 0,1 \
  --timestamp vlfm_A0_r1

# VLFM（无实例核验）
bash scripts/run_instance_ablation.sh \
  E1 \
  --vlm qwen3.5-flash \
  --nav-base vlfm \
  --suite avs5_vlfm_20261123 \
  --gpus 0,1 \
  --timestamp vlfm_E1_r1
