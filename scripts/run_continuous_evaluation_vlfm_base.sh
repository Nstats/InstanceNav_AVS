#!/usr/bin/env bash
# 本机 InstanceNav val10 固定 100 题。单组失败继续，失败组最后手动补跑。
set +e
cd "$(dirname "$0")/.." || exit 1

bash scripts/launch_vlfm_servers.sh --gpu 0 || exit $?

# RA：先筛选再主动走近；准入不计票，近处另问一次确认。
bash scripts/run_instance_ablation.sh RA \
  --nav-base vlfm \
  --vlm qwen3.5-flash \
  --instance-verification-prompt match_visible_evidence \
  --enable-candidate-approach true \
  --verify-before-candidate true \
  --instance-confirmation-count-unit frame \
  --instance-confirmation-threshold 1 \
  --gpus 0,1 \
  --suite vlfm_inst100_visible_selection \
  --timestamp inst100_vlfm_visible_ra_local_r1

# B：不预筛，主动走近；近处一次 yes 即确认。
bash scripts/run_instance_ablation.sh B \
  --nav-base vlfm \
  --vlm qwen3.5-flash \
  --instance-verification-prompt match_visible_evidence \
  --enable-candidate-approach true \
  --verify-before-candidate false \
  --instance-confirmation-count-unit frame \
  --instance-confirmation-threshold 1 \
  --gpus 0,1 \
  --suite vlfm_inst100_visible_selection \
  --timestamp inst100_vlfm_visible_b_local_r1

# E5：在 B 上只改票数；同一候选在两个不同策略步得到 yes 才确认。
bash scripts/run_instance_ablation.sh E5 \
  --nav-base vlfm \
  --vlm qwen3.5-flash \
  --instance-verification-prompt match_visible_evidence \
  --enable-candidate-approach true \
  --verify-before-candidate false \
  --instance-confirmation-count-unit frame \
  --instance-confirmation-threshold 2 \
  --gpus 0,1 \
  --suite vlfm_inst100_visible_selection \
  --timestamp inst100_vlfm_visible_e5_local_r1
