#!/usr/bin/env bash
# E0: CoIN A0 baseline on the 100 frozen val_unseen episodes (scripts/run_coin.sh defaults).
# A0 = unconfirmed candidates never become the navigation goal; one "yes" within 4 m locks.
# Base fixes stay off (goal point frozen at lock, depth gate off).
#
#   bash scripts/run_coin_e0.sh --timestamp coin_e0_strict
#   bash scripts/run_coin_e0.sh --instance-verification-prompt match_context_optional --timestamp coin_e0_ctx
#
# Any run_coin.sh flag may follow; later flags override the preset.
set -euo pipefail
exec bash "$(dirname "$0")/run_coin.sh" \
  --enable-candidate-approach false \
  --instance-verification-prompt match_strict_no \
  --goal-refresh-after-lock false \
  --depth-valid-fraction-gate 0 \
  "$@"
