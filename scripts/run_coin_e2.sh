#!/usr/bin/env bash
# E2: E0 (CoIN A0) + base fixes, on the 100 frozen val_unseen episodes.
#   goal refresh after lock: GLIP keeps running after a lock and the goal point moves to the
#     nearest goal-category box within 1.5 m (no model call, the locked instance is kept);
#   depth gate: a box whose unoccluded centre has < 50% valid depth gets no distance
#     (mirrors, glass), and the distance is the median of the valid pixels.
#
#   bash scripts/run_coin_e2.sh --timestamp coin_e2_strict
#   bash scripts/run_coin_e2.sh --instance-verification-prompt match --timestamp coin_e2_match
#
# Any run_coin.sh flag may follow; later flags override the preset.
set -euo pipefail
exec bash "$(dirname "$0")/run_coin.sh" \
  --enable-candidate-approach false \
  --instance-verification-prompt match_strict_no \
  --goal-refresh-after-lock true \
  --goal-refresh-radius-m 1.5 \
  --depth-valid-fraction-gate 0.5 \
  "$@"
