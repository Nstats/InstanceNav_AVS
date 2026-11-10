#!/usr/bin/env bash
# Evaluate the fixed 100-episode InstanceNav sample.
# Instance confirmation applies before setting candidate or confirmed goal states.
# Choose the primary confirmation policy, then optionally recheck positive matches.
set -euo pipefail

# Usage: bash scripts/run_instance100.sh --gpus 0,1 --timestamp NAME
: "${CUDA_VISIBLE_DEVICES:=0}"

# A same-named environment variable sets the starting value; a flag overrides it.
: "${LANDMARK_COMPARISON_REPEATS:=1}"  # --landmark-comparison-repeats
: "${LANDMARK_COMPARISON_WORKERS:=10}"  # --landmark-comparison-workers
# 仅控制地标搜索提示词；实例核验始终使用完整目标描述。
# true：类别＋完整实例描述引导探索。false：类别引导探索＋完整描述实例核验。
: "${ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION:=false}"  # --enable-instance-description-for-exploration
: "${ENABLE_OBJECT_RELATIONS:=false}"  # --enable-object-relations
: "${ENABLE_GROUP_LLM_FILTER:=true}"  # --enable-group-llm-filter
: "${ENABLE_ROOM_LLM_FILTER:=true}"  # --enable-room-llm-filter
# 实例核验总开关；不控制组/房间 LLM 筛选。旧环境变量仅作为兼容回退。
: "${ENABLE_INSTANCE_CONFIRMATION:=${ENABLE_INSTANCE_CONFIRMATION_BEFORE_FIND_GOAL:-true}}"  # --enable-instance-confirmation
unset ENABLE_INSTANCE_CONFIRMATION_BEFORE_FIND_GOAL
# 同一候选的正式核验有效通过次数；所有类别统一，达到阈值即锁定。总开关关闭时不生效。
: "${INSTANCE_CONFIRMATION_THRESHOLD:=1}"  # --instance-confirmation-threshold
: "${INSTANCE_CONFIRMATION_COUNT_UNIT:=detection}"  # --instance-confirmation-count-unit
# 独立的候选准入核验；准入结果不计锁定票，近距离锁定核验另行调用。
: "${VERIFY_BEFORE_CANDIDATE:=false}"  # --verify-before-candidate
# INSTANCE_CONFIRMATION_MODE (--instance-confirmation-mode) 的两种选择（实例确认总开关开启时生效）：
#   positive_match：正面确认（默认）；只有 yes 保留，no/unknown 排除。
#   contradiction：用显著冲突检测替代正面确认；yes 表示有冲突并排除，no/unknown 保留。
: "${INSTANCE_CONFIRMATION_MODE:=positive_match}"  # --instance-confirmation-mode
# INSTANCE_VERIFICATION_PROMPT (--instance-verification-prompt)：positive_match 核验提示词的版本，“是”的写法两版相同。
#   match：原始提示词（默认），证据不足才答 unknown；实际上几乎从不答 unknown，“否”里混着“看不清”。
#   match_strict_no：只有明确矛盾才答 no；看不清、被遮挡、属性或周围环境不在画面里都答 unknown。
: "${INSTANCE_VERIFICATION_PROMPT:=match_strict_no}"  # --instance-verification-prompt
: "${INSTANCE_VERIFICATION_SCOPE:=instance}"  # --instance-verification-scope
: "${INSTANCE_REJECTION_POLICY:=none}"  # --instance-rejection-policy
: "${INSTANCE_VERIFICATION_RECORD_INPUTS:=false}"  # --instance-verification-record-inputs
# 独立复验开关：仅用于 positive_match，且要求实例确认总开关开启。
# true：正式确认前，将当前及同一候选的全部历史正面观测一次性多图复验。
# false：不做这次额外复验（默认）；与上面的 contradiction 主确认方式不同。
: "${ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK:=false}"  # --enable-post-confirmation-contradiction-check
# 仅用于开启实例确认的 positive_match，与后置冲突复验开关独立。
# true：受困时保留各候选正计数，解除导航锁定；实际脱困并重新确认同一候选后才能再次锁定。
# false：受困重置时清空正计数，保持历史流程（默认）。
: "${PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY:=false}"  # --preserve-positive-evidence-on-recovery
# 只采集、不停止（实验计划.md S6），仅用于采集离线回放数据，不是导航方法；要求 positive_match。
# true：到了原本会锁定的时候只记录 [Intended Lock]，该候选不再接近，继续探索直到步数用完。
# false：正常锁定并停止（默认）。
: "${INSTANCE_COLLECT_ONLY:=false}"  # --instance-collect-only
# 相对证据决策：默认关闭；开启后延迟锁定，用 q/U 决定何时返回，再比较候选。
: "${ENABLE_RELATIVE_EVIDENCE:=false}"  # --enable-relative-evidence
: "${RELATIVE_EVIDENCE_CALIBRATION:=configs/relative_evidence_s6.json}"
: "${RELATIVE_EVIDENCE_USE_U:=true}"
: "${RELATIVE_EVIDENCE_EARLY_STOP:=true}"
: "${RELATIVE_EVIDENCE_PAIRWISE:=true}"
: "${RELATIVE_EVIDENCE_DROP_REJECTED:=true}"
: "${RAG_MODE:=0}"  # --rag-mode
# VLM (--vlm) 的选择：
#   qwen3.5-flash：当前默认；走 .env 里的 DASHSCOPE 兼容接口。
#   gemini-2.5-flash：走 OpenRouter，模型名为 google/gemini-2.5-flash，凭据是 OPENROUTER_API_KEY。
#   qwen3-vl-8b-instruct：走硅基流动，模型名为 Qwen/Qwen3-VL-8B-Instruct，凭据是 SILICONFLOW_API_KEY。
: "${VLM:=qwen3.5-flash}"  # --vlm

# 0: preflight then navigate. --check-only sets 1 and exits after validation.
check_only=0
passthrough=()
user_timestamp=""
gpu_list=""
worker_gpu=""
work_queue=""
resume=0

usage() {
  cat <<EOF
Usage: bash scripts/run_instance100.sh [options]

Defaults to two worker processes on GPUs 0 and 1; --gpus overrides the pair.
Same-named environment variables set the defaults below.
Flags override those environment variables.

Options:
  --landmark-comparison-repeats N   Positive integer (default: 1)
  --landmark-comparison-workers N   Positive integer (default: 10)
  --enable-instance-description-for-exploration BOOL
                                    true/false/1/0/yes/no (default: false)
                                    false: category-guided exploration; verification keeps full descriptions
  --enable-object-relations BOOL    true/false/1/0/yes/no (default: false)
  --enable-group-llm-filter BOOL    true/false/1/0/yes/no (default: true)
  --enable-room-llm-filter BOOL     true/false/1/0/yes/no (default: true)
  --enable-instance-confirmation BOOL
                                    true/false/1/0/yes/no (default: true)
                                    Master switch for instance checks (group/room filters independent)
  --instance-confirmation-threshold N
                                    Required passing verification count for every category (default: 1)
                                    Applies only when instance confirmation is enabled
  --instance-confirmation-count-unit NAME
                                    detection (legacy default) or frame (one vote per candidate/frame)
  --verify-before-candidate BOOL    Independent candidate-admission check (default: false)
                                    Admission passes never count as confirmation votes
  --instance-confirmation-mode MODE
                                    positive_match (default) or contradiction
                                    Select the primary instance confirmation policy
  --instance-verification-prompt NAME
                                    match or match_strict_no (default)
                                    match_strict_no answers no only for a clear contradiction
  --instance-verification-scope NAME
                                    instance (default) or category
  --instance-rejection-policy NAME  none (default), no, or non_yes; episode-local permanent rejection
  --instance-verification-record-inputs BOOL
                                    Save exact request/image evidence (default: false)
  --enable-post-confirmation-contradiction-check BOOL
                                    true/false/1/0/yes/no (default: false)
                                    Recheck a positive match using all its positive views
                                    Requires positive_match and instance confirmation
  --rag-mode N                      0=off, 1=collect, 2=use, 3=full (default: 0)
  --vlm NAME                        qwen3.5-flash (default), gemini-2.5-flash, or qwen3-vl-8b-instruct
                                    qwen3.5-flash uses the current compatible API
                                    gemini-2.5-flash uses OpenRouter
                                    qwen3-vl-8b-instruct uses SiliconFlow
  --preserve-positive-evidence-on-recovery BOOL
                                    true/false/1/0/yes/no (default: false)
                                    Keep candidate evidence through navigation recovery
                                    Requires positive_match and instance confirmation
  --instance-collect-only BOOL      true/false/1/0/yes/no (default: false)
                                    Data collection only: log intended locks, never stop early
                                    Requires positive_match, instance confirmation, no post check
  --timestamp NAME                  Result directory id; required again with --resume
  --enable-relative-evidence BOOL   Relative-evidence stopping and candidate comparison (default: false)
  --relative-evidence-calibration PATH
                                    Frozen model (default: configs/relative_evidence_s6.json)
  --relative-evidence-use-u BOOL    Include unseen candidate mass (default: true)
  --relative-evidence-early-stop BOOL
                                    Use P1 before policy step 860 (default: true)
  --relative-evidence-pairwise BOOL Compare up to three yes candidates (default: true)
  --relative-evidence-drop-rejected BOOL
                                    Exclude no candidates from competition (default: true)
                                    Requires S6 settings: confirmation on, positive_match, k=1,
                                    admission/post-check/preservation/collect-only off, RAG=0
  --resume                          Continue the same timestamp from the next episode
  --gpus A,B                        Run the same 100 tasks on two GPUs (default: 0,1). Each GPU keeps
                                    one process, claims the next unfinished task, and
                                    writes its own log
  --check-only                      Validate inputs without navigation
  -h, --help                        Show this help
EOF
}

require_option_value() {
  local flag="$1"
  if (( $# < 2 )) || [[ -z "${2:-}" || "${2}" == --* ]]; then
    echo "Missing value for ${flag}" >&2
    exit 2
  fi
}

require_positive_int() {
  local name="$1"
  local raw="${!name}"
  if ! [[ "${raw}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer, got: ${raw}" >&2
    exit 2
  fi
}

require_rag_mode() {
  local name="$1"
  local raw="${!name}"
  if ! [[ "${raw}" =~ ^[0-3]$ ]]; then
    echo "${name} must be 0, 1, 2, or 3, got: ${raw}" >&2
    exit 2
  fi
}

normalize_bool_var() {
  local name="$1"
  local raw="${!name}"
  local lower="${raw,,}"
  case "${lower}" in
    true|1|yes) printf -v "${name}" '%s' true ;;
    false|0|no) printf -v "${name}" '%s' false ;;
    *)
      echo "${name} must be true/false/1/0/yes/no, got: ${raw}" >&2
      exit 2
      ;;
  esac
}

while (( $# > 0 )); do
  case "$1" in
    --check-only)
      check_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --landmark-comparison-repeats)
      require_option_value "$1" "${2:-}"
      LANDMARK_COMPARISON_REPEATS="$2"
      shift 2
      ;;
    --landmark-comparison-workers)
      require_option_value "$1" "${2:-}"
      LANDMARK_COMPARISON_WORKERS="$2"
      shift 2
      ;;
    --enable-instance-description-for-exploration)
      require_option_value "$1" "${2:-}"
      ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION="$2"
      shift 2
      ;;
    --enable-object-relations)
      require_option_value "$1" "${2:-}"
      ENABLE_OBJECT_RELATIONS="$2"
      shift 2
      ;;
    --enable-group-llm-filter)
      require_option_value "$1" "${2:-}"
      ENABLE_GROUP_LLM_FILTER="$2"
      shift 2
      ;;
    --enable-room-llm-filter)
      require_option_value "$1" "${2:-}"
      ENABLE_ROOM_LLM_FILTER="$2"
      shift 2
      ;;
    --enable-instance-confirmation|--enable-instance-confirmation-before-find-goal)
      require_option_value "$1" "${2:-}"
      ENABLE_INSTANCE_CONFIRMATION="$2"
      shift 2
      ;;
    --instance-confirmation-threshold)
      require_option_value "$1" "${2:-}"
      INSTANCE_CONFIRMATION_THRESHOLD="$2"
      shift 2
      ;;
    --verify-before-candidate)
      require_option_value "$1" "${2:-}"
      VERIFY_BEFORE_CANDIDATE="$2"
      shift 2
      ;;
    --instance-confirmation-mode)
      require_option_value "$1" "${2:-}"
      INSTANCE_CONFIRMATION_MODE="$2"
      shift 2
      ;;
    --instance-verification-prompt)
      require_option_value "$1" "${2:-}"
      INSTANCE_VERIFICATION_PROMPT="$2"
      shift 2
      ;;
    --instance-verification-scope|--instance-rejection-policy|--instance-verification-record-inputs|--instance-confirmation-count-unit)
      require_option_value "$1" "${2:-}"
      ablation_name="${1#--}"
      ablation_name="${ablation_name//-/_}"
      printf -v "${ablation_name^^}" '%s' "$2"
      shift 2
      ;;
    --enable-post-confirmation-contradiction-check)
      require_option_value "$1" "${2:-}"
      ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK="$2"
      shift 2
      ;;
    --enable-contradiction-check)
      echo '--enable-contradiction-check is retired; use --instance-confirmation-mode contradiction (old true) or positive_match (old false)' >&2
      exit 2
      ;;
    --preserve-positive-evidence-on-recovery)
      require_option_value "$1" "${2:-}"
      PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY="$2"
      shift 2
      ;;
    --instance-collect-only)
      require_option_value "$1" "${2:-}"
      INSTANCE_COLLECT_ONLY="$2"
      shift 2
      ;;
    --rag-mode)
      require_option_value "$1" "${2:-}"
      RAG_MODE="$2"
      shift 2
      ;;
    --enable-relative-evidence|--relative-evidence-calibration|--relative-evidence-use-u|--relative-evidence-early-stop|--relative-evidence-pairwise|--relative-evidence-drop-rejected)
      require_option_value "$1" "${2:-}"
      relative_name="${1#--}"
      relative_name="${relative_name//-/_}"
      printf -v "${relative_name^^}" '%s' "$2"
      shift 2
      ;;
    --vlm)
      require_option_value "$1" "${2:-}"
      VLM="$2"
      shift 2
      ;;
    --timestamp)
      require_option_value "$1" "${2:-}"
      user_timestamp="$2"
      shift 2
      ;;
    --resume)
      resume=1
      passthrough+=(--resume)
      shift
      ;;
    --gpus)
      require_option_value "$1" "${2:-}"
      gpu_list="$2"
      shift 2
      ;;
    --worker-gpu)
      require_option_value "$1" "${2:-}"
      worker_gpu="$2"
      shift 2
      ;;
    --work-queue)
      require_option_value "$1" "${2:-}"
      work_queue="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

require_positive_int LANDMARK_COMPARISON_REPEATS
require_positive_int LANDMARK_COMPARISON_WORKERS
normalize_bool_var ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION
normalize_bool_var ENABLE_OBJECT_RELATIONS
normalize_bool_var ENABLE_GROUP_LLM_FILTER
normalize_bool_var ENABLE_ROOM_LLM_FILTER
normalize_bool_var ENABLE_INSTANCE_CONFIRMATION
normalize_bool_var VERIFY_BEFORE_CANDIDATE
require_positive_int INSTANCE_CONFIRMATION_THRESHOLD
if [[ "${ENABLE_CONTRADICTION_CHECK+x}" == x ]]; then
  echo 'ENABLE_CONTRADICTION_CHECK is retired; use INSTANCE_CONFIRMATION_MODE=contradiction (old true) or positive_match (old false)' >&2
  exit 2
fi
INSTANCE_CONFIRMATION_MODE="${INSTANCE_CONFIRMATION_MODE,,}"
case "$INSTANCE_CONFIRMATION_MODE" in
  positive_match|contradiction) ;;
  *) echo 'INSTANCE_CONFIRMATION_MODE must be positive_match or contradiction' >&2; exit 2 ;;
esac
INSTANCE_VERIFICATION_PROMPT="${INSTANCE_VERIFICATION_PROMPT,,}"
case "$INSTANCE_VERIFICATION_PROMPT" in
  match|match_strict_no) ;;
  *) echo 'INSTANCE_VERIFICATION_PROMPT must be match or match_strict_no' >&2; exit 2 ;;
esac
normalize_bool_var ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK
normalize_bool_var PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY
normalize_bool_var INSTANCE_COLLECT_ONLY
normalize_bool_var INSTANCE_VERIFICATION_RECORD_INPUTS
normalize_bool_var ENABLE_RELATIVE_EVIDENCE
normalize_bool_var RELATIVE_EVIDENCE_USE_U
normalize_bool_var RELATIVE_EVIDENCE_EARLY_STOP
normalize_bool_var RELATIVE_EVIDENCE_PAIRWISE
normalize_bool_var RELATIVE_EVIDENCE_DROP_REJECTED
if [[ "$ENABLE_RELATIVE_EVIDENCE" == true ]]; then
  relative_conflicts=()
  [[ "$ENABLE_INSTANCE_CONFIRMATION" == true ]] || relative_conflicts+=(ENABLE_INSTANCE_CONFIRMATION=true)
  [[ "$INSTANCE_CONFIRMATION_MODE" == positive_match ]] || relative_conflicts+=(INSTANCE_CONFIRMATION_MODE=positive_match)
  [[ "$VERIFY_BEFORE_CANDIDATE" == false ]] || relative_conflicts+=(VERIFY_BEFORE_CANDIDATE=false)
  [[ "$INSTANCE_CONFIRMATION_THRESHOLD" == 1 ]] || relative_conflicts+=(INSTANCE_CONFIRMATION_THRESHOLD=1)
  [[ "$ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK" == false ]] || relative_conflicts+=(ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK=false)
  [[ "$PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY" == false ]] || relative_conflicts+=(PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY=false)
  [[ "$INSTANCE_COLLECT_ONLY" == false ]] || relative_conflicts+=(INSTANCE_COLLECT_ONLY=false)
  [[ "$RAG_MODE" == 0 ]] || relative_conflicts+=(RAG_MODE=0)
  if (( ${#relative_conflicts[@]} )); then
    echo "Relative evidence requires: ${relative_conflicts[*]}" >&2
    exit 2
  fi
fi
if [[ "$ENABLE_INSTANCE_CONFIRMATION" == true && "$ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK" == true && "$INSTANCE_CONFIRMATION_MODE" != positive_match ]]; then
  echo 'ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK requires INSTANCE_CONFIRMATION_MODE=positive_match and ENABLE_INSTANCE_CONFIRMATION=true' >&2
  exit 2
fi
require_rag_mode RAG_MODE
case "$VLM" in
  qwen3.5-flash|gemini-2.5-flash|qwen3-vl-8b-instruct) ;;
  *) echo "VLM must be qwen3.5-flash, gemini-2.5-flash, or qwen3-vl-8b-instruct, got: ${VLM}" >&2; exit 2 ;;
esac
if [[ "$ENABLE_INSTANCE_CONFIRMATION" == true && "$PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY" == true && "$INSTANCE_CONFIRMATION_MODE" != positive_match ]]; then
  echo 'PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY requires INSTANCE_CONFIRMATION_MODE=positive_match and ENABLE_INSTANCE_CONFIRMATION=true' >&2
  exit 2
fi
if [[ "$INSTANCE_COLLECT_ONLY" == true && ( "$ENABLE_INSTANCE_CONFIRMATION" != true || "$INSTANCE_CONFIRMATION_MODE" != positive_match ) ]]; then
  echo 'INSTANCE_COLLECT_ONLY requires ENABLE_INSTANCE_CONFIRMATION=true and INSTANCE_CONFIRMATION_MODE=positive_match' >&2
  exit 2
fi
if [[ "$INSTANCE_COLLECT_ONLY" == true && "$ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK" == true ]]; then
  echo 'INSTANCE_COLLECT_ONLY cannot be combined with ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK' >&2
  exit 2
fi
# Apply the default only to the launcher, never to its single-GPU workers.
if [[ -z "${gpu_list}" && -z "${worker_gpu}" ]]; then
  gpu_list="0,1"
fi
if [[ -n "${gpu_list}" && -n "${worker_gpu}" ]]; then
  echo '--gpus cannot be combined with --worker-gpu' >&2
  exit 2
fi
if [[ -n "${worker_gpu}" ]]; then
  if [[ -z "${work_queue}" || -z "${user_timestamp}" ]]; then
    echo '--worker-gpu requires --work-queue and --timestamp' >&2
    exit 2
  fi
  if ! [[ "${worker_gpu}" =~ ^[0-9]+$ ]]; then
    echo "--worker-gpu must be a non-negative integer, got: ${worker_gpu}" >&2
    exit 2
  fi
fi
if [[ -n "${gpu_list}" ]]; then
  if ! [[ "${gpu_list}" =~ ^[0-9]+,[0-9]+$ ]]; then
    echo '--gpus expects two ids such as 0,1' >&2
    exit 2
  fi
  IFS=',' read -r _gpu_a _gpu_b <<<"${gpu_list}"
  if [[ "${_gpu_a}" == "${_gpu_b}" ]]; then
    echo '--gpus ids must differ' >&2
    exit 2
  fi
fi

export LANDMARK_COMPARISON_REPEATS
export LANDMARK_COMPARISON_WORKERS
export ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION
export ENABLE_OBJECT_RELATIONS
export ENABLE_GROUP_LLM_FILTER
export ENABLE_ROOM_LLM_FILTER
export ENABLE_INSTANCE_CONFIRMATION
export INSTANCE_CONFIRMATION_THRESHOLD
export INSTANCE_CONFIRMATION_COUNT_UNIT
export VERIFY_BEFORE_CANDIDATE
export INSTANCE_CONFIRMATION_MODE
export INSTANCE_VERIFICATION_PROMPT
export INSTANCE_VERIFICATION_SCOPE INSTANCE_REJECTION_POLICY INSTANCE_VERIFICATION_RECORD_INPUTS
export ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK
export PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY
export INSTANCE_COLLECT_ONLY
export ENABLE_RELATIVE_EVIDENCE RELATIVE_EVIDENCE_CALIBRATION RELATIVE_EVIDENCE_USE_U
export RELATIVE_EVIDENCE_EARLY_STOP RELATIVE_EVIDENCE_PAIRWISE RELATIVE_EVIDENCE_DROP_REJECTED
export RAG_MODE
export VLM
case "$VLM" in
  qwen3.5-flash)
    export QWEN_MODEL_NAME=qwen3.5-flash
    export PSGNAV_MODEL_BACKEND=openai
    ;;
  gemini-2.5-flash)
    export QWEN_MODEL_NAME=google/gemini-2.5-flash
    export PSGNAV_MODEL_BACKEND=openrouter
    ;;
  qwen3-vl-8b-instruct)
    export QWEN_MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct
    export PSGNAV_MODEL_BACKEND=siliconflow
    ;;
esac

script_path="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$script_path")/.."
psgnav_python="${PSGNAV_PYTHON:-$(command -v python || true)}"
if [[ ! -x "$psgnav_python" ]]; then
  echo '请先按 DEPLOYMENT_ZH.md 配置 Python 环境，再通过 PSGNAV_PYTHON 指定解释器。' >&2
  exit 1
fi
relative_evidence_header='{"enabled":false}'
if [[ "$ENABLE_RELATIVE_EVIDENCE" == true ]]; then
  relative_evidence_header="$("$psgnav_python" - <<'PY'
import json, os
from configs.navigation_config import relative_evidence_run_header
print(json.dumps(relative_evidence_run_header(dataset='instancenav', backend=os.environ['PSGNAV_MODEL_BACKEND'])))
PY
)"
fi
"$psgnav_python" - <<'PY'
from configs.navigation_config import validate_instance_ablation
validate_instance_ablation()
import inspect, torch
if 'weights_only' not in inspect.signature(torch.load).parameters:
    raise SystemExit('当前 Torch 不支持本版使用的 weights_only 参数；请使用复现指南中的 PyTorch 2.7 环境。')
PY

export PSGNAV_ENV_FILE="$PWD/.env"
export PSGNAV_API_CONCURRENCY=8
export PSGNAV_API_MAX_TOKENS=4096
export PSGNAV_API_CACHE_SIZE=4096
export PSGNAV_EXACT_CACHE=1
export PSGNAV_TF32=1
export PSGNAV_TORCH_THREADS=4
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES
export HF_HOME="$PWD/data/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NLTK_DATA="$PWD/data/nltk_data${NLTK_DATA:+:$NLTK_DATA}"
export PYTHONPATH="$PWD/habitat-lab:$PWD:$PWD/GLIP${PYTHONPATH:+:$PYTHONPATH}"

if (( check_only )); then
  "$psgnav_python" tools/preflight_instancenav.py --config configs/runtime/instancenav_unseen10_1000atomic_1m.yaml
  echo "LANDMARK_COMPARISON_REPEATS: ${LANDMARK_COMPARISON_REPEATS}"
  echo "LANDMARK_COMPARISON_WORKERS: ${LANDMARK_COMPARISON_WORKERS}"
  echo "ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION: ${ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION}"
  echo "ENABLE_OBJECT_RELATIONS: ${ENABLE_OBJECT_RELATIONS}"
  echo "ENABLE_GROUP_LLM_FILTER: ${ENABLE_GROUP_LLM_FILTER}"
  echo "ENABLE_ROOM_LLM_FILTER: ${ENABLE_ROOM_LLM_FILTER}"
  echo "ENABLE_INSTANCE_CONFIRMATION: ${ENABLE_INSTANCE_CONFIRMATION}"
  echo "INSTANCE_CONFIRMATION_THRESHOLD: ${INSTANCE_CONFIRMATION_THRESHOLD}"
  echo "INSTANCE_CONFIRMATION_COUNT_UNIT: ${INSTANCE_CONFIRMATION_COUNT_UNIT}"
  echo "VERIFY_BEFORE_CANDIDATE: ${VERIFY_BEFORE_CANDIDATE}"
  echo "INSTANCE_CONFIRMATION_MODE: ${INSTANCE_CONFIRMATION_MODE}"
  echo "INSTANCE_VERIFICATION_PROMPT: ${INSTANCE_VERIFICATION_PROMPT}"
  echo "INSTANCE_VERIFICATION_SCOPE: ${INSTANCE_VERIFICATION_SCOPE}"
  echo "INSTANCE_REJECTION_POLICY: ${INSTANCE_REJECTION_POLICY}"
  echo "INSTANCE_VERIFICATION_RECORD_INPUTS: ${INSTANCE_VERIFICATION_RECORD_INPUTS}"
  echo "ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK: ${ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK}"
  echo "PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY: ${PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY}"
  echo "INSTANCE_COLLECT_ONLY: ${INSTANCE_COLLECT_ONLY}"
  echo "ENABLE_RELATIVE_EVIDENCE: ${ENABLE_RELATIVE_EVIDENCE}"
  echo "Relative evidence: ${relative_evidence_header}"
  echo "RAG_MODE: ${RAG_MODE}"
  echo "VLM: ${VLM}"
  echo "QWEN_MODEL_NAME: ${QWEN_MODEL_NAME}"
  echo "PSGNAV_MODEL_BACKEND: ${PSGNAV_MODEL_BACKEND}"
  echo "Preflight passed."
  exit 0
fi

run_worker() {
  local log_file tasks_file exit_status worker_pid
  local -a cmd
  run_timestamp="${user_timestamp}"
  mkdir -p outputs
  log_file="$PWD/outputs/instancenav_val10_100_${run_timestamp}_gpu${worker_gpu}.log"
  tasks_file="$PWD/outputs/instancenav_val10_100_${run_timestamp}_gpu${worker_gpu}.tasks.json"
  {
    echo "GPU: ${worker_gpu}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "LANDMARK_COMPARISON_REPEATS: ${LANDMARK_COMPARISON_REPEATS}"
    echo "LANDMARK_COMPARISON_WORKERS: ${LANDMARK_COMPARISON_WORKERS}"
    echo "ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION: ${ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION}"
    echo "ENABLE_OBJECT_RELATIONS: ${ENABLE_OBJECT_RELATIONS}"
    echo "ENABLE_GROUP_LLM_FILTER: ${ENABLE_GROUP_LLM_FILTER}"
    echo "ENABLE_ROOM_LLM_FILTER: ${ENABLE_ROOM_LLM_FILTER}"
    echo "ENABLE_INSTANCE_CONFIRMATION: ${ENABLE_INSTANCE_CONFIRMATION}"
    echo "INSTANCE_CONFIRMATION_THRESHOLD: ${INSTANCE_CONFIRMATION_THRESHOLD}"
    echo "INSTANCE_CONFIRMATION_COUNT_UNIT: ${INSTANCE_CONFIRMATION_COUNT_UNIT}"
    echo "VERIFY_BEFORE_CANDIDATE: ${VERIFY_BEFORE_CANDIDATE}"
    echo "INSTANCE_CONFIRMATION_MODE: ${INSTANCE_CONFIRMATION_MODE}"
    echo "INSTANCE_VERIFICATION_PROMPT: ${INSTANCE_VERIFICATION_PROMPT}"
    echo "INSTANCE_VERIFICATION_SCOPE: ${INSTANCE_VERIFICATION_SCOPE}"
    echo "INSTANCE_REJECTION_POLICY: ${INSTANCE_REJECTION_POLICY}"
    echo "INSTANCE_VERIFICATION_RECORD_INPUTS: ${INSTANCE_VERIFICATION_RECORD_INPUTS}"
    echo "ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK: ${ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK}"
    echo "PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY: ${PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY}"
    echo "INSTANCE_COLLECT_ONLY: ${INSTANCE_COLLECT_ONLY}"
    echo "ENABLE_RELATIVE_EVIDENCE: ${ENABLE_RELATIVE_EVIDENCE}"
    echo "Relative evidence: ${relative_evidence_header}"
    echo "RAG_MODE: ${RAG_MODE}"
    echo "VLM: ${VLM}"
    echo "QWEN_MODEL_NAME: ${QWEN_MODEL_NAME}"
    echo "PSGNAV_MODEL_BACKEND: ${PSGNAV_MODEL_BACKEND}"
    echo "Log: ${log_file}"
    echo "Tasks: ${tasks_file}"
    echo "Results: ${PWD}/data/results_instancenav_${run_timestamp}/gpu${worker_gpu}/"
    echo "Work queue: ${work_queue}"
    echo "Dataset indices follow the single-GPU order, starting at 0."
    echo "This process claims the next unfinished index from the shared queue."
    echo "Completed tasks are listed below as [Schedule] lines and in ${tasks_file}."
  } >>"${log_file}"
  cmd=(
    "$psgnav_python" -u PSG_Nav.py
    --dataset instancenav --dataset_split val
    --config configs/runtime/instancenav_unseen10_1000atomic_1m.yaml
    --rag_mode "${RAG_MODE}" --episode_start 0 --episode_end 100
    --timestamp "${run_timestamp}"
    --work-queue "${work_queue}" --worker-gpu "${worker_gpu}"
  )
  if (( resume )); then
    cmd+=(--resume)
  fi
  set +e
  unset PSGNAV_PROGRESS_FD
  PSGNAV_LOG_FILE="${log_file}" PSGNAV_TASKS_FILE="${tasks_file}" \
    "${cmd[@]}" >>"${log_file}" 2>&1 &
  worker_pid=$!
  # The coordinator signals this shell; forward termination to the evaluator.
  trap 'kill "${worker_pid}" 2>/dev/null || true' INT TERM
  wait "${worker_pid}"
  exit_status=$?
  if kill -0 "${worker_pid}" 2>/dev/null; then
    wait "${worker_pid}"
    exit_status=$?
  fi
  trap - INT TERM
  set -e
  if (( exit_status != 0 )); then
    echo "Evaluation interrupted (exit ${exit_status}). Log: ${log_file}" >>"${log_file}"
  fi
  exit "${exit_status}"
}

run_dual() {
  local gpu_a gpu_b results_root queue log_a log_b started watch_pid
  local status_a=0 status_b=0 merge_status=0 preflight_status=0
  local -a worker_common
  IFS=',' read -r gpu_a gpu_b <<<"${gpu_list}"
  if [[ -n "${user_timestamp}" ]]; then
    run_timestamp="${user_timestamp}"
  else
    run_timestamp="$(date +%Y%m%d_%H%M%S)_$$"
  fi
  results_root="$PWD/data/results_instancenav_${run_timestamp}"
  queue="${results_root}/schedule.json"
  log_a="$PWD/outputs/instancenav_val10_100_${run_timestamp}_gpu${gpu_a}.log"
  log_b="$PWD/outputs/instancenav_val10_100_${run_timestamp}_gpu${gpu_b}.log"
  mkdir -p outputs "${results_root}"
  echo "Dual GPU: ${gpu_a} and ${gpu_b}"
  echo "Timestamp: ${run_timestamp}"
  echo "LANDMARK_COMPARISON_REPEATS: ${LANDMARK_COMPARISON_REPEATS}"
  echo "LANDMARK_COMPARISON_WORKERS: ${LANDMARK_COMPARISON_WORKERS}"
  echo "ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION: ${ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION}"
  echo "ENABLE_OBJECT_RELATIONS: ${ENABLE_OBJECT_RELATIONS}"
  echo "ENABLE_GROUP_LLM_FILTER: ${ENABLE_GROUP_LLM_FILTER}"
  echo "ENABLE_ROOM_LLM_FILTER: ${ENABLE_ROOM_LLM_FILTER}"
  echo "ENABLE_INSTANCE_CONFIRMATION: ${ENABLE_INSTANCE_CONFIRMATION}"
  echo "INSTANCE_CONFIRMATION_THRESHOLD: ${INSTANCE_CONFIRMATION_THRESHOLD}"
  echo "INSTANCE_CONFIRMATION_COUNT_UNIT: ${INSTANCE_CONFIRMATION_COUNT_UNIT}"
  echo "VERIFY_BEFORE_CANDIDATE: ${VERIFY_BEFORE_CANDIDATE}"
  echo "INSTANCE_CONFIRMATION_MODE: ${INSTANCE_CONFIRMATION_MODE}"
  echo "INSTANCE_VERIFICATION_PROMPT: ${INSTANCE_VERIFICATION_PROMPT}"
  echo "INSTANCE_VERIFICATION_SCOPE: ${INSTANCE_VERIFICATION_SCOPE}"
  echo "INSTANCE_REJECTION_POLICY: ${INSTANCE_REJECTION_POLICY}"
  echo "INSTANCE_VERIFICATION_RECORD_INPUTS: ${INSTANCE_VERIFICATION_RECORD_INPUTS}"
  echo "ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK: ${ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK}"
  echo "PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY: ${PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY}"
  echo "INSTANCE_COLLECT_ONLY: ${INSTANCE_COLLECT_ONLY}"
  echo "ENABLE_RELATIVE_EVIDENCE: ${ENABLE_RELATIVE_EVIDENCE}"
  echo "Relative evidence: ${relative_evidence_header}"
  echo "RAG_MODE: ${RAG_MODE}"
  echo "VLM: ${VLM}"
  echo "QWEN_MODEL_NAME: ${QWEN_MODEL_NAME}"
  echo "PSGNAV_MODEL_BACKEND: ${PSGNAV_MODEL_BACKEND}"
  echo "Log gpu${gpu_a}: ${log_a}"
  echo "Log gpu${gpu_b}: ${log_b}"
  echo "Results: ${results_root}/"
  echo "Queue: ${queue}"
  set +e
  "$psgnav_python" tools/preflight_instancenav.py --config configs/runtime/instancenav_unseen10_1000atomic_1m.yaml
  preflight_status=$?
  set -e
  if (( preflight_status != 0 )); then
    echo "Preflight failed (exit ${preflight_status})."
    exit "${preflight_status}"
  fi
  if [[ -e "${queue}" || -d "${results_root}/gpu${gpu_a}" || -d "${results_root}/gpu${gpu_b}" || -d "${results_root}/experiment_0" ]]; then
    if (( ! resume )); then
      echo 'Results already exist; choose a new timestamp or pass --resume' >&2
      exit 1
    fi
    if [[ ! -f "${queue}" ]]; then
      echo "--resume requires ${queue}" >&2
      exit 1
    fi
    "$psgnav_python" tools/eval_schedule.py requeue --schedule "${queue}"
  elif (( resume )); then
    echo "--resume requires the original timestamp and ${queue}" >&2
    exit 1
  else
    "$psgnav_python" tools/eval_schedule.py init --schedule "${queue}" --total 100 --max-episode-steps 1000
  fi
  # fd 3 stays on the terminal. Workers do not inherit the progress variable.
  exec 3>&1
  export PSGNAV_PROGRESS_FD=3
  started="$("$psgnav_python" -c 'import time; print(f"{time.time():.3f}")')"
  "$psgnav_python" -u tools/eval_schedule.py watch \
    --schedule "${queue}" --gpus "${gpu_a},${gpu_b}" --started "${started}" &
  watch_pid=$!
  worker_common=(--work-queue "${queue}" --timestamp "${run_timestamp}")
  if (( resume )); then
    worker_common+=(--resume)
  fi
  set +e
  dual_pid_a=""
  dual_pid_b=""
  trap '[[ -n "${dual_pid_a:-}" ]] && kill "${dual_pid_a}" 2>/dev/null || true; [[ -n "${dual_pid_b:-}" ]] && kill "${dual_pid_b}" 2>/dev/null || true' INT TERM
  CUDA_VISIBLE_DEVICES="${gpu_a}" bash "${script_path}" --worker-gpu "${gpu_a}" "${worker_common[@]}" &
  dual_pid_a=$!
  CUDA_VISIBLE_DEVICES="${gpu_b}" bash "${script_path}" --worker-gpu "${gpu_b}" "${worker_common[@]}" &
  dual_pid_b=$!
  # Observe whichever worker exits first. A fatal model/format error must stop
  # this run, even when the failing worker is the second one in the list.
  if ! wait -n "${dual_pid_a}" "${dual_pid_b}"; then
    echo 'A worker failed; stopping both workers of this evaluation. See the GPU logs for the error.' >&2
    kill "${dual_pid_a}" "${dual_pid_b}" 2>/dev/null || true
  fi
  wait "${dual_pid_a}"
  status_a=$?
  if kill -0 "${dual_pid_a}" 2>/dev/null; then
    wait "${dual_pid_a}"
    status_a=$?
  fi
  wait "${dual_pid_b}"
  status_b=$?
  if kill -0 "${dual_pid_b}" 2>/dev/null; then
    wait "${dual_pid_b}"
    status_b=$?
  fi
  trap - INT TERM
  dual_pid_a=""
  dual_pid_b=""
  set -e
  "$psgnav_python" tools/eval_schedule.py stop-watch --schedule "${queue}" || true
  wait "${watch_pid}" || true
  set +e
  "$psgnav_python" tools/eval_schedule.py merge \
    --results-root "${results_root}" --gpus "${gpu_a},${gpu_b}" --total 100
  merge_status=$?
  set -e
  echo "gpu${gpu_a} exit ${status_a}; gpu${gpu_b} exit ${status_b}; merge exit ${merge_status}"
  echo "Log gpu${gpu_a}: ${log_a}"
  echo "Log gpu${gpu_b}: ${log_b}"
  if (( status_a != 0 || status_b != 0 || merge_status != 0 )); then
    echo "Evaluation incomplete."
    exit 1
  fi
  exit 0
}

if [[ -n "${gpu_list}" ]]; then
  run_dual
fi
if [[ -n "${worker_gpu}" ]]; then
  run_worker
fi

if [[ -n "${user_timestamp}" ]]; then
  run_timestamp="${user_timestamp}"
else
  run_timestamp="$(date +%Y%m%d_%H%M%S)_$$"
fi
passthrough+=(--timestamp "${run_timestamp}")

mkdir -p outputs
log_file="$PWD/outputs/instancenav_val10_100_${run_timestamp}.log"
# fd 3 stays on the terminal for the progress line. Python text goes to the log.
exec 3>&1
{
  echo "GPU: ${CUDA_VISIBLE_DEVICES}"
  echo "LANDMARK_COMPARISON_REPEATS: ${LANDMARK_COMPARISON_REPEATS}"
  echo "LANDMARK_COMPARISON_WORKERS: ${LANDMARK_COMPARISON_WORKERS}"
  echo "ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION: ${ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION}"
  echo "ENABLE_OBJECT_RELATIONS: ${ENABLE_OBJECT_RELATIONS}"
  echo "ENABLE_GROUP_LLM_FILTER: ${ENABLE_GROUP_LLM_FILTER}"
  echo "ENABLE_ROOM_LLM_FILTER: ${ENABLE_ROOM_LLM_FILTER}"
  echo "ENABLE_INSTANCE_CONFIRMATION: ${ENABLE_INSTANCE_CONFIRMATION}"
  echo "INSTANCE_CONFIRMATION_THRESHOLD: ${INSTANCE_CONFIRMATION_THRESHOLD}"
  echo "INSTANCE_CONFIRMATION_COUNT_UNIT: ${INSTANCE_CONFIRMATION_COUNT_UNIT}"
  echo "VERIFY_BEFORE_CANDIDATE: ${VERIFY_BEFORE_CANDIDATE}"
  echo "INSTANCE_CONFIRMATION_MODE: ${INSTANCE_CONFIRMATION_MODE}"
  echo "INSTANCE_VERIFICATION_PROMPT: ${INSTANCE_VERIFICATION_PROMPT}"
  echo "INSTANCE_VERIFICATION_SCOPE: ${INSTANCE_VERIFICATION_SCOPE}"
  echo "INSTANCE_REJECTION_POLICY: ${INSTANCE_REJECTION_POLICY}"
  echo "INSTANCE_VERIFICATION_RECORD_INPUTS: ${INSTANCE_VERIFICATION_RECORD_INPUTS}"
  echo "ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK: ${ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK}"
  echo "PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY: ${PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY}"
  echo "INSTANCE_COLLECT_ONLY: ${INSTANCE_COLLECT_ONLY}"
  echo "ENABLE_RELATIVE_EVIDENCE: ${ENABLE_RELATIVE_EVIDENCE}"
  echo "Relative evidence: ${relative_evidence_header}"
  echo "RAG_MODE: ${RAG_MODE}"
  echo "VLM: ${VLM}"
  echo "QWEN_MODEL_NAME: ${QWEN_MODEL_NAME}"
  echo "PSGNAV_MODEL_BACKEND: ${PSGNAV_MODEL_BACKEND}"
} | tee -a "${log_file}" >&3
set +e
{
  echo "Log: ${log_file}"
  echo "Results: ${PWD}/data/results_instancenav_${run_timestamp}/"
  echo "Key parameters and the progress line stay on the terminal. Everything else goes to this log."
  "$psgnav_python" tools/preflight_instancenav.py --config configs/runtime/instancenav_unseen10_1000atomic_1m.yaml
} >>"${log_file}" 2>&1
preflight_status=$?
set -e
if (( preflight_status != 0 )); then
  echo "Preflight failed (exit ${preflight_status}). Log: ${log_file}" >&3
  exit "${preflight_status}"
fi

set +e
PSGNAV_LOG_FILE="${log_file}" PSGNAV_PROGRESS_FD=3 "$psgnav_python" -u PSG_Nav.py --dataset instancenav --dataset_split val \
  --config configs/runtime/instancenav_unseen10_1000atomic_1m.yaml \
  --rag_mode "${RAG_MODE}" --episode_start 0 --episode_end 100 \
  "${passthrough[@]}" >>"${log_file}" 2>&1
exit_status=$?
set -e
if (( exit_status != 0 )); then
  echo "Evaluation interrupted (exit ${exit_status}). Log: ${log_file}" >&3
  exit "${exit_status}"
fi
