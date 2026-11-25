#!/usr/bin/env bash
# H20：InstanceNav val10 固定 100 题，服务和导航都用 GPU 0。
set +e
cd "$(dirname "$0")/.." || exit 1

bash scripts/launch_vlfm_servers.sh --gpu 0 || exit $?

# RA 第二轮：与本机 RA 第一轮配置相同；先预筛，近处再独立确认一次。
bash scripts/run_instance_ablation.sh RA \
  --nav-base vlfm \
  --vlm qwen3.5-flash \
  --instance-verification-prompt match_visible_evidence \
  --enable-candidate-approach true \
  --verify-before-candidate true \
  --instance-confirmation-count-unit frame \
  --instance-confirmation-threshold 1 \
  --gpus 0 \
  --suite vlfm_inst100_visible_selection \
  --timestamp inst100_vlfm_visible_ra_h20_r2
