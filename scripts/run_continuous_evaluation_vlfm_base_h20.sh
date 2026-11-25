#!/usr/bin/env bash
# H20：只跑 CoIN val_seen 全集；服务和导航都用 GPU 0。
set +e
cd "$(dirname "$0")/.." || exit 1

export GROUNDING_DINO_PORT="${GROUNDING_DINO_PORT:-22181}"
export BLIP2ITM_PORT="${BLIP2ITM_PORT:-22182}"
export SAM_PORT="${SAM_PORT:-22183}"
export YOLOV7_PORT="${YOLOV7_PORT:-22184}"
bash scripts/launch_vlfm_servers.sh --gpu 0 --session instancenav_h20 || exit $?

# CoIN val_seen 全集：831 题，A0 主配置。
bash scripts/run_coin_e0.sh \
  --split val_seen \
  --full \
  --nav-base vlfm \
  --vlm qwen3.5-flash \
  --instance-verification-prompt match_visible_evidence \
  --enable-instance-confirmation true \
  --gpu 0 \
  --timestamp coin_val_seen_full_vlfm_a0_visible_h20_r1
