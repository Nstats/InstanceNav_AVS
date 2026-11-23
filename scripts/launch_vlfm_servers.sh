#!/usr/bin/env bash
# Start/reuse the four local VLFM endpoints; --stop affects only this launcher's children.
# Usage: bash scripts/launch_vlfm_servers.sh [--gpu N] [--session NAME] [--stop | --check-only]
set -euo pipefail
CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
unset PYTHONPATH
exec "${VLFM_PYTHON:-${PSGNAV_PYTHON:-python}}" tools/vlfm_servers.py "$@"
