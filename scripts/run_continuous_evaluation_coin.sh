#!/usr/bin/env bash

bash scripts/run_coin_e0.sh \
  --split val_seen \
  --subset 100 \
  --gpus 0,1 \
  --nav-base psgnav \
  --vlm qwen3.5-flash \
  --instance-verification-prompt match_visible_evidence \
  --timestamp coin_seen100_visible_evidence_local_r1

bash scripts/run_coin_e0.sh \
  --split val_seen_synonyms \
  --subset 100 \
  --gpus 0,1 \
  --nav-base psgnav \
  --vlm qwen3.5-flash \
  --instance-verification-prompt match_visible_evidence \
  --timestamp coin_syn100_visible_evidence_local_r1

bash scripts/run_coin_e0.sh \
  --split val_unseen \
  --subset 100 \
  --gpus 0,1 \
  --nav-base psgnav \
  --vlm qwen3.5-flash \
  --instance-verification-prompt match_visible_evidence \
  --timestamp coin_unseen100_visible_evidence_local_r1
