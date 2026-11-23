#!/usr/bin/env bash
# A0VisibleEvidence：固定 val10 的 100 个任务，使用 match_visible_evidence 提示词。
bash scripts/run_instance_ablation.sh \
  A0VisibleEvidence \
  --gpus 0 \
  --nav-base psgnav \
  --vlm qwen3.5-flash \
  --suite avs_global_prompt \
  --timestamp inst100_visible_evidence_h20_r1
