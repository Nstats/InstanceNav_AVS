#!/usr/bin/env bash
# One explicit experiment per invocation; never launches the full suite.
set -euo pipefail
ablation_root="$(cd "$(dirname "$0")/.." && pwd)"
ablation_python="${PSGNAV_PYTHON:-$(command -v python)}"
exec "$ablation_python" "$ablation_root/tools/run_instance_ablation.py" "$@"
