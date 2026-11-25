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

# 全集主配置 A0：沿用百题的 VLFM、可见证据提示词和单票确认；每组各一轮。
# InstanceNav 官方 val 全集：1,000 题。
bash scripts/run_instance_ablation.sh A0 \
  --full \
  --nav-base vlfm \
  --vlm qwen3.5-flash \
  --instance-verification-prompt match_visible_evidence \
  --gpus 0,1 \
  --suite vlfm_full_visible \
  --timestamp inst_val_full_vlfm_a0_visible_local_r1

# CoIN val_unseen 全集：459 题。
bash scripts/run_coin_e0.sh \
  --split val_unseen \
  --full \
  --nav-base vlfm \
  --vlm qwen3.5-flash \
  --instance-verification-prompt match_visible_evidence \
  --enable-instance-confirmation true \
  --gpus 0,1 \
  --timestamp coin_val_unseen_full_vlfm_a0_visible_local_r1

# CoIN val_seen_synonyms 全集：359 题。
bash scripts/run_coin_e0.sh \
  --split val_seen_synonyms \
  --full \
  --nav-base vlfm \
  --vlm qwen3.5-flash \
  --instance-verification-prompt match_visible_evidence \
  --enable-instance-confirmation true \
  --gpus 0,1 \
  --timestamp coin_val_seen_synonyms_full_vlfm_a0_visible_local_r1
