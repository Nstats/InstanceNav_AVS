#!/usr/bin/env bash
# Evaluate PSG-Nav on the official CoIN-Bench sensor/task protocol:
# 500 atomic steps, success = STOP within 0.25 m of a target view point, full instruction,
# no user interaction, CoIN's own RGB (42 deg) / depth (79 deg) geometry.
# Restore the 20261108 CoIN A policy: description exploration and the match prompt.
# InstanceNav's separate launcher/configuration is not changed.
#
# The default evaluation is a fixed 100-episode val_unseen subset tracked in git
# (configs/coin_val_unseen_subset100_seed77.json: all 47 targets, then 53 more episodes, seed 77).
# Each split has such a frozen 100-episode pool (configs/coin_<split>_subset100_seed77.json,
# same rule and seed: val_seen all 86 targets + 14, val_seen_synonyms all 36 targets + 64);
# every subset of a split is drawn from its pool, --full runs the whole split.
#
#   bash scripts/run_coin.sh                    # default: the 100 frozen val_unseen episodes, GPUs 0,1
#   bash scripts/run_coin.sh --subset 20        # 20 of the 100 frozen episodes
#   bash scripts/run_coin.sh --full             # whole val_unseen split (459 episodes)
#   bash scripts/run_coin.sh --split val_seen   # the 100 frozen val_seen episodes
#   bash scripts/run_coin.sh --subset 3 --gpu 2 # one sequential process (smoke runs)
set -euo pipefail

# CoIN A defaults; the other shared switches still match run_instance100.sh.
: "${LANDMARK_COMPARISON_REPEATS:=1}"
: "${LANDMARK_COMPARISON_WORKERS:=10}"
: "${ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION:=true}"
: "${ENABLE_OBJECT_RELATIONS:=false}"
: "${ENABLE_GROUP_LLM_FILTER:=true}"
: "${ENABLE_ROOM_LLM_FILTER:=true}"
: "${ENABLE_INSTANCE_CONFIRMATION:=true}"
: "${INSTANCE_CONFIRMATION_THRESHOLD:=1}"
: "${VERIFY_BEFORE_CANDIDATE:=false}"
: "${INSTANCE_CONFIRMATION_MODE:=positive_match}"
: "${INSTANCE_VERIFICATION_PROMPT:=match}"
: "${ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK:=false}"
: "${PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY:=false}"
: "${INSTANCE_COLLECT_ONLY:=false}"
: "${ENABLE_RELATIVE_EVIDENCE:=false}"
: "${RAG_MODE:=0}"
: "${VLM:=qwen3.5-flash}"
# A0 (paper main configuration): false = unconfirmed candidates never become the
# navigation goal; they are verified only when exploration passes within range.
: "${ENABLE_CANDIDATE_APPROACH:=true}"
# Base fixes (E2), all off by default = the original policy.
# GOAL_REFRESH_AFTER_LOCK: keep running GLIP after a lock and move the goal point to the
# nearest goal-category box within GOAL_REFRESH_RADIUS_M of it (no model call).
# DEPTH_VALID_FRACTION_GATE: a box whose unoccluded centre has less valid depth than this
# fraction gets no distance (mirrors, glass); 0 disables it.
: "${GOAL_REFRESH_AFTER_LOCK:=false}"
: "${GOAL_REFRESH_RADIUS_M:=1.5}"
: "${DEPTH_VALID_FRACTION_GATE:=0}"
# Object-navigation base (--nav-base): psgnav (default, PSG_Nav.py, unchanged) or vlfm
# (VLFM_Nav.py; start scripts/launch_vlfm_servers.sh first). VLFM ignores the PSG-Nav-only
# switches (description exploration, relations, group/room filters, base fixes); with
# instance confirmation on it runs AVS A0 only (--enable-candidate-approach false).
: "${NAV_BASE:=psgnav}"
# Rejection ablations (same switches as scripts/run_instance100.sh); none = no/unknown
# answers leave no trace (AVS). no / non_yes exclude the candidate for the episode (E3 / E4).
: "${INSTANCE_REJECTION_POLICY:=none}"
# instance (AVS): verify the full description; category: the VLM checks the category only (E2).
: "${INSTANCE_VERIFICATION_SCOPE:=instance}"
: "${INSTANCE_REJECTION_SCOPE:=episode}"

# Defaults: the 100 frozen val_unseen episodes on GPUs 0,1. A same-named environment
# variable sets the starting value; a flag overrides it. --subset 0 / --full runs
# the whole split.
: "${COIN_SPLIT:=val_unseen}"
: "${COIN_SUBSET:=100}"
: "${COIN_SEED:=77}"
# Frozen 100-episode pools (tracked); a split's subsets are drawn from its pool only.
val_unseen_pool="configs/coin_val_unseen_subset100_seed77.json"
val_seen_pool="configs/coin_val_seen_subset100_seed77.json"
val_seen_synonyms_pool="configs/coin_val_seen_synonyms_subset100_seed77.json"
split="$COIN_SPLIT"
subset="$COIN_SUBSET"
seed="$COIN_SEED"
config="configs/coin_contextnav.yaml"
gpu_list=""
single_gpu=""
worker_gpu=""
work_queue=""
user_timestamp=""
resume=0
check_only=0

usage() {
  cat <<EOF
Usage: bash scripts/run_coin.sh [options]

Runs with no options: the 100 frozen val_unseen episodes on GPUs 0,1
(${val_unseen_pool}: all 47 targets, then 53 more episodes, seed 77).

  --split S          val_seen | val_seen_synonyms | val_unseen (default: val_unseen)
  --subset N         100 (default): read the tracked manifest directly; no sampling.
                     1..99: draw a smaller subset of the frozen episodes, distinct targets
                     first. 0: run the whole split. Tracked manifests:
                     ${val_unseen_pool}, ${val_seen_pool},
                     ${val_seen_synonyms_pool}.
  --full             Run the whole split (same as --subset 0; val_unseen 459, val_seen 831,
                     val_seen_synonyms 359 episodes)
  --seed N           Subset seed (default: 77)
  --config PATH      Habitat config (default: configs/coin_contextnav.yaml; ignored with --subset)
  --gpus A,B         Two workers sharing one task queue (default: 0,1)
  --gpu N            One sequential process on GPU N (smoke runs)
  --timestamp NAME   Result id (default: <split>_<date>_<pid>); required again with --resume
  --resume           Continue the same timestamp
  --check-only       Run tools/preflight_coin.py and exit
  --nav-base NAME    psgnav (default, PSG_Nav.py) or vlfm (VLFM_Nav.py; needs
                     scripts/launch_vlfm_servers.sh). vlfm + instance confirmation = AVS A0
                     and requires --enable-candidate-approach false; with
                     --enable-instance-confirmation false it is the original VLFM

Navigation switches take the same flags as scripts/run_instance100.sh, for example
  --enable-instance-description-for-exploration BOOL   (default: true)
  --instance-verification-prompt match|match_strict_no|match_visible_evidence|match_context_optional (default: match)
  --enable-instance-confirmation BOOL                  (default: true)
  --instance-confirmation-mode positive_match|contradiction
  --verify-before-candidate BOOL, --instance-confirmation-threshold N, --rag-mode N
  --enable-object-relations / --enable-group-llm-filter / --enable-room-llm-filter BOOL
  --enable-post-confirmation-contradiction-check / --preserve-positive-evidence-on-recovery BOOL
  --instance-collect-only BOOL, --landmark-comparison-repeats N, --landmark-comparison-workers N
  --vlm qwen3.5-flash|gemini-2.5-flash|qwen3-vl-8b-instruct
  --enable-candidate-approach BOOL  Unconfirmed candidates become the navigation goal
                                    (default: true; false = A0)
  --instance-rejection-policy none|no|non_yes  (default: none) Exclude a candidate for the
                                    episode after a "no" (no, E3) or any non-yes answer (non_yes, E4)
  --instance-rejection-scope episode|approach  (default: episode)
  --instance-verification-scope instance|category  (default: instance; category = E2)
  --goal-refresh-after-lock BOOL    Move the locked goal point to its newest estimate (default: false)
  --goal-refresh-radius-m M         Largest refresh jump in metres, (0, 5] (default: 1.5)
  --depth-valid-fraction-gate F     Minimum valid-depth fraction of a box centre, [0, 1);
                                    0 = off (default: 0)
EOF
}

switch_flags=" landmark-comparison-repeats landmark-comparison-workers enable-instance-description-for-exploration
  enable-object-relations enable-group-llm-filter enable-room-llm-filter enable-instance-confirmation
  instance-confirmation-threshold verify-before-candidate instance-confirmation-mode instance-verification-prompt
  enable-post-confirmation-contradiction-check preserve-positive-evidence-on-recovery
  instance-collect-only rag-mode vlm enable-candidate-approach goal-refresh-after-lock
  goal-refresh-radius-m depth-valid-fraction-gate instance-rejection-policy instance-rejection-scope
  instance-verification-scope "
switch_flags=" $(echo $switch_flags) "

need_value() {
  if (( $2 < 2 )) || [[ -z "${3:-}" || "${3}" == --* ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while (( $# > 0 )); do
  flag="$1"
  case "$flag" in
    -h|--help) usage; exit 0 ;;
    --check-only) check_only=1; shift ;;
    --resume) resume=1; shift ;;
    --full) subset=0; shift ;;
    --split) need_value "$flag" $# "${2:-}"; split="$2"; shift 2 ;;
    --subset) need_value "$flag" $# "${2:-}"; subset="$2"; shift 2 ;;
    --seed) need_value "$flag" $# "${2:-}"; seed="$2"; shift 2 ;;
    --config) need_value "$flag" $# "${2:-}"; config="$2"; shift 2 ;;
    --gpus) need_value "$flag" $# "${2:-}"; gpu_list="$2"; shift 2 ;;
    --gpu) need_value "$flag" $# "${2:-}"; single_gpu="$2"; shift 2 ;;
    --worker-gpu) need_value "$flag" $# "${2:-}"; worker_gpu="$2"; shift 2 ;;
    --work-queue) need_value "$flag" $# "${2:-}"; work_queue="$2"; shift 2 ;;
    --timestamp) need_value "$flag" $# "${2:-}"; user_timestamp="$2"; shift 2 ;;
    --nav-base) need_value "$flag" $# "${2:-}"; NAV_BASE="$2"; shift 2 ;;
    --enable-relative-evidence|--relative-evidence-*)
      echo 'Relative evidence is calibrated on InstanceNav only and is not available for CoIN.' >&2
      exit 2 ;;
    --*)
      name="${flag#--}"
      if [[ "$switch_flags" != *" $name "* ]]; then
        echo "Unknown argument: $flag" >&2
        exit 2
      fi
      need_value "$flag" $# "${2:-}"
      name="${name//-/_}"
      printf -v "${name^^}" '%s' "$2"
      shift 2 ;;
    *) echo "Unknown argument: $flag" >&2; exit 2 ;;
  esac
done

fail() { echo "$*" >&2; exit 2; }
normalize_bool() {
  local name="$1" raw="${!1}"
  case "${raw,,}" in
    true|1|yes) printf -v "$name" '%s' true ;;
    false|0|no) printf -v "$name" '%s' false ;;
    *) fail "$name must be true/false/1/0/yes/no, got: $raw" ;;
  esac
}
case "$split" in
  val_seen|val_seen_synonyms|val_unseen) ;;
  *) fail '--split must be val_seen, val_seen_synonyms or val_unseen' ;;
esac
for name in ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION ENABLE_OBJECT_RELATIONS ENABLE_GROUP_LLM_FILTER \
    ENABLE_ROOM_LLM_FILTER ENABLE_INSTANCE_CONFIRMATION VERIFY_BEFORE_CANDIDATE \
    ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY \
    INSTANCE_COLLECT_ONLY ENABLE_RELATIVE_EVIDENCE ENABLE_CANDIDATE_APPROACH GOAL_REFRESH_AFTER_LOCK; do
  normalize_bool "$name"
done
[[ "$GOAL_REFRESH_RADIUS_M" =~ ^[0-9]+(\.[0-9]+)?$ ]] && awk -v v="$GOAL_REFRESH_RADIUS_M" 'BEGIN{exit !(v > 0 && v <= 5)}' \
  || fail "GOAL_REFRESH_RADIUS_M must be in (0, 5], got: $GOAL_REFRESH_RADIUS_M"
[[ "$DEPTH_VALID_FRACTION_GATE" =~ ^[0-9]+(\.[0-9]+)?$ ]] && awk -v v="$DEPTH_VALID_FRACTION_GATE" 'BEGIN{exit !(v >= 0 && v < 1)}' \
  || fail "DEPTH_VALID_FRACTION_GATE must be in [0, 1), got: $DEPTH_VALID_FRACTION_GATE"
[[ "$ENABLE_RELATIVE_EVIDENCE" == false ]] || fail 'ENABLE_RELATIVE_EVIDENCE must stay false for CoIN'
for name in LANDMARK_COMPARISON_REPEATS LANDMARK_COMPARISON_WORKERS INSTANCE_CONFIRMATION_THRESHOLD; do
  [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || fail "$name must be a positive integer, got: ${!name}"
done
[[ "$RAG_MODE" =~ ^[0-3]$ ]] || fail "RAG_MODE must be 0-3, got: $RAG_MODE"
INSTANCE_CONFIRMATION_MODE="${INSTANCE_CONFIRMATION_MODE,,}"
INSTANCE_VERIFICATION_PROMPT="${INSTANCE_VERIFICATION_PROMPT,,}"
case "$INSTANCE_VERIFICATION_PROMPT" in
  match|match_strict_no|match_visible_evidence|match_context_optional) ;;
  *) fail 'INSTANCE_VERIFICATION_PROMPT must be match, match_strict_no, match_visible_evidence or match_context_optional' ;;
esac
INSTANCE_REJECTION_POLICY="${INSTANCE_REJECTION_POLICY,,}"
INSTANCE_REJECTION_SCOPE="${INSTANCE_REJECTION_SCOPE,,}"
case "$INSTANCE_REJECTION_POLICY" in
  none|no|non_yes) ;;
  *) fail 'INSTANCE_REJECTION_POLICY must be none, no or non_yes' ;;
esac
INSTANCE_VERIFICATION_SCOPE="${INSTANCE_VERIFICATION_SCOPE,,}"
case "$INSTANCE_VERIFICATION_SCOPE" in
  instance) ;;
  category)
    [[ "$ENABLE_INSTANCE_CONFIRMATION" == true && "$INSTANCE_CONFIRMATION_MODE" == positive_match ]] \
      || fail 'INSTANCE_VERIFICATION_SCOPE=category requires instance confirmation and positive_match' ;;
  *) fail 'INSTANCE_VERIFICATION_SCOPE must be instance or category' ;;
esac
case "$INSTANCE_REJECTION_SCOPE" in
  episode|approach) ;;
  *) fail 'INSTANCE_REJECTION_SCOPE must be episode or approach' ;;
esac
if [[ "$INSTANCE_REJECTION_POLICY" != none ]] && [[ "$ENABLE_INSTANCE_CONFIRMATION" != true \
    || "$INSTANCE_CONFIRMATION_MODE" != positive_match ]]; then
  fail 'INSTANCE_REJECTION_POLICY requires instance confirmation and positive_match'
fi
if [[ "$INSTANCE_REJECTION_SCOPE" == approach && "$INSTANCE_REJECTION_POLICY" == none ]]; then
  fail 'INSTANCE_REJECTION_SCOPE=approach requires INSTANCE_REJECTION_POLICY no or non_yes'
fi
[[ "$INSTANCE_CONFIRMATION_MODE" == positive_match || "$INSTANCE_CONFIRMATION_MODE" == contradiction ]] \
  || fail 'INSTANCE_CONFIRMATION_MODE must be positive_match or contradiction'
if [[ "$INSTANCE_CONFIRMATION_MODE" != positive_match || "$ENABLE_INSTANCE_CONFIRMATION" != true ]]; then
  for name in ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY INSTANCE_COLLECT_ONLY; do
    [[ "${!name}" == false ]] || fail "$name requires INSTANCE_CONFIRMATION_MODE=positive_match and ENABLE_INSTANCE_CONFIRMATION=true"
  done
fi
if [[ "$ENABLE_CANDIDATE_APPROACH" == false ]] && [[ "$ENABLE_INSTANCE_CONFIRMATION" != true \
    || "$INSTANCE_CONFIRMATION_MODE" != positive_match || "$INSTANCE_COLLECT_ONLY" != false ]]; then
  fail 'ENABLE_CANDIDATE_APPROACH=false requires instance confirmation, positive_match and collect-only off'
fi
if [[ "$INSTANCE_COLLECT_ONLY" == true && "$ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK" == true ]]; then
  fail 'INSTANCE_COLLECT_ONLY cannot be combined with ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK'
fi
case "$NAV_BASE" in
  psgnav) nav_entry=PSG_Nav.py ;;
  vlfm)
    nav_entry=VLFM_Nav.py
    if [[ "$ENABLE_INSTANCE_CONFIRMATION" == true && "$ENABLE_CANDIDATE_APPROACH" != false ]]; then
      fail 'NAV_BASE=vlfm with instance confirmation runs AVS A0 only: pass --enable-candidate-approach false'
    fi ;;
  *) fail "NAV_BASE must be psgnav or vlfm, got: $NAV_BASE" ;;
esac
case "$VLM" in
  qwen3.5-flash) export QWEN_MODEL_NAME=qwen3.5-flash PSGNAV_MODEL_BACKEND=openai ;;
  gemini-2.5-flash) export QWEN_MODEL_NAME=google/gemini-2.5-flash PSGNAV_MODEL_BACKEND=openrouter ;;
  qwen3-vl-8b-instruct) export QWEN_MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct PSGNAV_MODEL_BACKEND=siliconflow ;;
  *) fail "VLM must be qwen3.5-flash, gemini-2.5-flash, or qwen3-vl-8b-instruct, got: $VLM" ;;
esac
if [[ "$subset" == 0 ]]; then subset=""; fi
if [[ -n "$subset" && ! "$subset" =~ ^[1-9][0-9]*$ ]]; then fail '--subset expects a non-negative integer (0 = whole split)'; fi
if [[ -n "$gpu_list" && -n "$single_gpu" ]]; then fail '--gpus and --gpu are exclusive'; fi
if [[ -z "$gpu_list" && -z "$single_gpu" && -z "$worker_gpu" ]]; then gpu_list="0,1"; fi
if [[ -n "$gpu_list" ]]; then
  [[ "$gpu_list" =~ ^[0-9]+,[0-9]+$ ]] || fail '--gpus expects two ids such as 0,1'
  IFS=',' read -r gpu_a gpu_b <<<"$gpu_list"
  [[ "$gpu_a" != "$gpu_b" ]] || fail '--gpus ids must differ'
fi
if [[ -n "$single_gpu" && ! "$single_gpu" =~ ^[0-9]+$ ]]; then fail '--gpu expects one id such as 2'; fi
if [[ -n "$worker_gpu" && ( -z "$work_queue" || -z "$user_timestamp" ) ]]; then
  fail '--worker-gpu requires --work-queue and --timestamp'
fi
if (( resume )) && [[ -z "$user_timestamp" ]]; then fail '--resume requires --timestamp'; fi

export LANDMARK_COMPARISON_REPEATS LANDMARK_COMPARISON_WORKERS ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION
export ENABLE_OBJECT_RELATIONS ENABLE_GROUP_LLM_FILTER ENABLE_ROOM_LLM_FILTER ENABLE_INSTANCE_CONFIRMATION
export INSTANCE_CONFIRMATION_THRESHOLD VERIFY_BEFORE_CANDIDATE INSTANCE_CONFIRMATION_MODE INSTANCE_VERIFICATION_PROMPT
export ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY INSTANCE_COLLECT_ONLY
export ENABLE_RELATIVE_EVIDENCE RAG_MODE VLM ENABLE_CANDIDATE_APPROACH
export GOAL_REFRESH_AFTER_LOCK GOAL_REFRESH_RADIUS_M DEPTH_VALID_FRACTION_GATE NAV_BASE
export INSTANCE_REJECTION_POLICY INSTANCE_REJECTION_SCOPE INSTANCE_VERIFICATION_SCOPE
unset ENABLE_INSTANCE_CONFIRMATION_BEFORE_FIND_GOAL

script_path="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$script_path")/.."
psgnav_python="${PSGNAV_PYTHON:-$(command -v python || true)}"
[[ -x "$psgnav_python" ]] || { echo 'Set PSGNAV_PYTHON to the psgnav-instance interpreter.' >&2; exit 1; }
# Same runtime environment as scripts/run_instance100.sh.
export PSGNAV_ENV_FILE="$PWD/.env"
export PSGNAV_API_CONCURRENCY=8 PSGNAV_API_MAX_TOKENS=4096 PSGNAV_API_CACHE_SIZE=4096
export PSGNAV_EXACT_CACHE=1 PSGNAV_TF32=1 PSGNAV_TORCH_THREADS=4 PYTHONUNBUFFERED=1
export HF_HOME="$PWD/data/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NLTK_DATA="$PWD/data/nltk_data${NLTK_DATA:+:$NLTK_DATA}"
export PYTHONPATH="$PWD/habitat-lab:$PWD:$PWD/GLIP${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$NAV_BASE" == vlfm ]]; then
  "$psgnav_python" -m nav_bases.vlfm_runtime check --group navigation
fi

switch_summary() {
  local name
  for name in LANDMARK_COMPARISON_REPEATS LANDMARK_COMPARISON_WORKERS ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION \
      ENABLE_OBJECT_RELATIONS ENABLE_GROUP_LLM_FILTER ENABLE_ROOM_LLM_FILTER ENABLE_INSTANCE_CONFIRMATION \
      INSTANCE_CONFIRMATION_THRESHOLD VERIFY_BEFORE_CANDIDATE INSTANCE_CONFIRMATION_MODE INSTANCE_VERIFICATION_PROMPT \
      ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY INSTANCE_COLLECT_ONLY \
      ENABLE_RELATIVE_EVIDENCE RAG_MODE VLM QWEN_MODEL_NAME PSGNAV_MODEL_BACKEND ENABLE_CANDIDATE_APPROACH \
      GOAL_REFRESH_AFTER_LOCK GOAL_REFRESH_RADIUS_M DEPTH_VALID_FRACTION_GATE NAV_BASE \
      INSTANCE_REJECTION_POLICY INSTANCE_REJECTION_SCOPE INSTANCE_VERIFICATION_SCOPE; do
    echo "${name}: ${!name}"
  done
}

# Subsets come from the split's frozen pool; the whole split (--full) uses none.
subset_pool=""
if [[ -n "$subset" ]]; then
  pool_var="${split}_pool"
  subset_pool="${!pool_var}"
fi

# Workers receive the launcher's config; only the launcher writes a subset.
if [[ -n "$subset" && -z "$worker_gpu" ]]; then
  subset_args=(--split "$split" --count "$subset" --pool "$subset_pool")
  if [[ "$subset" != 100 ]]; then subset_args+=(--seed "$seed"); fi
  subset_out="$("$psgnav_python" tools/coin_subset.py "${subset_args[@]}" 2>&1)" \
    || { echo "Subset generation failed: $(tail -n 5 <<<"$subset_out")" >&2; exit 1; }
  config="$(sed -n 's/^Config: //p' <<<"$subset_out")"
  [[ -n "$config" && -f "$config" ]] || { echo "Subset generation failed: $subset_out" >&2; exit 1; }
fi
preflight() { "$psgnav_python" tools/preflight_coin.py --config "$config" --split "$split" "$@"; }

if (( check_only )); then
  switch_summary
  echo "Split: $split"
  echo "Pool: ${subset_pool:-none}"
  echo "Config: $config"
  preflight
  exit $?
fi

if [[ -n "$user_timestamp" ]]; then
  run_timestamp="$user_timestamp"
else
  run_timestamp="${split}_$(date +%Y%m%d_%H%M%S)_$$"
fi
results_root="$PWD/data/results_coin_${run_timestamp}"
log_prefix="$PWD/outputs/coin_${run_timestamp}"
nav_cmd=("$psgnav_python" -u "$nav_entry" --dataset coin --dataset_split "$split" --config "$config"
         --rag_mode "$RAG_MODE" --timestamp "$run_timestamp")

run_worker() {
  local log_file="${log_prefix}_gpu${worker_gpu}.log" status pid
  local -a cmd=("${nav_cmd[@]}" --episode_start 0 --work-queue "$work_queue" --worker-gpu "$worker_gpu")
  (( resume )) && cmd+=(--resume)
  mkdir -p outputs
  { echo "GPU: ${worker_gpu}"; echo "Split: ${split}"; echo "Config: ${config}"; switch_summary
    echo "Results: ${results_root}/gpu${worker_gpu}/"; echo "Work queue: ${work_queue}"; } >>"$log_file"
  set +e
  unset PSGNAV_PROGRESS_FD
  PSGNAV_LOG_FILE="$log_file" PSGNAV_TASKS_FILE="${log_prefix}_gpu${worker_gpu}.tasks.json" \
    "${cmd[@]}" >>"$log_file" 2>&1 &
  pid=$!
  trap 'kill "$pid" 2>/dev/null || true' INT TERM
  wait "$pid"; status=$?
  if kill -0 "$pid" 2>/dev/null; then wait "$pid"; status=$?; fi
  trap - INT TERM
  (( status == 0 )) || echo "Evaluation interrupted (exit ${status}). Log: ${log_file}" >>"$log_file"
  exit "$status"
}

write_run_info() {
  mkdir -p "$results_root"
  RUN_INFO_SPLIT="$split" RUN_INFO_CONFIG="$config" RUN_INFO_SUBSET="$subset" RUN_INFO_SEED="$seed" \
  RUN_INFO_POOL="$subset_pool" RUN_INFO_TIMESTAMP="$run_timestamp" RUN_INFO_TOTAL="$1" RUN_INFO_MODE="$2" \
    "$psgnav_python" - "$results_root/run_info.json" <<'PY'
import hashlib, json, os, subprocess, sys
git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True).stdout
names = ('LANDMARK_COMPARISON_REPEATS LANDMARK_COMPARISON_WORKERS ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION '
         'ENABLE_OBJECT_RELATIONS ENABLE_GROUP_LLM_FILTER ENABLE_ROOM_LLM_FILTER ENABLE_INSTANCE_CONFIRMATION '
         'INSTANCE_CONFIRMATION_THRESHOLD VERIFY_BEFORE_CANDIDATE INSTANCE_CONFIRMATION_MODE '
        'ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY INSTANCE_COLLECT_ONLY '
         'ENABLE_RELATIVE_EVIDENCE RAG_MODE VLM QWEN_MODEL_NAME PSGNAV_MODEL_BACKEND '
         'INSTANCE_VERIFICATION_PROMPT PSGNAV_API_MAX_TOKENS PSGNAV_API_CACHE_SIZE PSGNAV_EXACT_CACHE '
         'ENABLE_CANDIDATE_APPROACH GOAL_REFRESH_AFTER_LOCK GOAL_REFRESH_RADIUS_M '
         'DEPTH_VALID_FRACTION_GATE NAV_BASE INSTANCE_REJECTION_POLICY INSTANCE_REJECTION_SCOPE '
         'INSTANCE_VERIFICATION_SCOPE').split()
info = {'dataset': 'coin', 'protocol': 'coin-official', 'nav_base': os.environ['NAV_BASE'], 'split': os.environ['RUN_INFO_SPLIT'],
        'config': os.environ['RUN_INFO_CONFIG'], 'subset': os.environ['RUN_INFO_SUBSET'] or None,
        'pool': os.environ['RUN_INFO_POOL'] or None,
        'seed': int(os.environ['RUN_INFO_SEED']), 'timestamp': os.environ['RUN_INFO_TIMESTAMP'],
        'total': int(os.environ['RUN_INFO_TOTAL']), 'mode': os.environ['RUN_INFO_MODE'],
        'git_commit': git('rev-parse', 'HEAD').strip(),
        'git_diff_sha256': hashlib.sha256(git('diff', 'HEAD').encode()).hexdigest(),
        'switches': {name: os.environ.get(name) for name in names}}
from pathlib import Path
info['evaluation_kind'] = 'collection' if os.environ['INSTANCE_COLLECT_ONLY'] == 'true' else 'navigation'
info['collection_schema_version'] = 2 if info['evaluation_kind'] == 'collection' else None
info['config_text'] = Path(info['config']).read_text()
from configs.dataset_categories import coin_policy_categories
import yaml
runtime_config = yaml.safe_load(info['config_text'])
manifest = runtime_config['DATASET'].get('SUBSET_MANIFEST')
if manifest:
    info['subset_manifest'] = manifest
    info['subset_manifest_sha256'] = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    if manifest == info['pool']:
        info['seed'] = None  # frozen episode IDs; no runtime sampling seed
info['policy_categories'] = coin_policy_categories()
task_config = runtime_config['TASK']
info['stop_rule'] = {'metric': task_config['STOP_DISTANCE_METRIC'],
                     'distance_m': task_config['STOP_DISTANCE']}
sources = ['PSG_Nav.py', 'utils/object_detection.py', 'utils/collection_recorder.py',
           'utils/collect_only.py', 'utils/instance_evidence.py', 'utils/instance_verifier.py',
           'utils/model_client.py', 'configs/navigation_config.py', 'configs/dataset_registry.py',
           'scripts/run_coin.sh', 'configs/coin_val_unseen_subset100_seed77.json',
           'configs/coin_val_seen_subset100_seed77.json', 'configs/coin_val_seen_synonyms_subset100_seed77.json',
           'configs/dataset_categories.py', 'graph/meta_func/io_func.py', 'graph/meta_func/nav_func.py',
           'utils/utils_sg_uq/merge_utils.py', 'utils/utils_fmm/fmm_planner.py']
if info['nav_base'] == 'vlfm':
    sources += ['VLFM_Nav.py', 'nav_bases/vlfm_agent.py', 'nav_bases/avs_decision.py',
                'nav_bases/vlfm_pointnav.py', 'utils/coin_geometry.py']
    from nav_bases.vlfm_runtime import UPSTREAM_REVISION
    info['vlfm_commit'] = UPSTREAM_REVISION  # upstream base, not a sibling working tree
    sources += ['nav_bases/vlfm_runtime.py', 'nav_bases/vlfm_server.py', 'nav_bases/vlfm_depth.py']
    sources += sorted(str(p) for p in Path('third_party/vlfm').rglob('*.py'))
info['source_sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
with open(sys.argv[1], 'w') as stream:
    json.dump(info, stream, indent=2)
PY
}

summarize() {
  "$psgnav_python" tools/summarize_coin.py "$results_root" || echo 'Summary failed; results are kept.' >&2
  if [[ "$INSTANCE_COLLECT_ONLY" == true ]]; then
    "$psgnav_python" tools/check_coin_collection.py "$results_root"
  fi
}

run_dual() {
  local queue="${results_root}/schedule.json" total started watch_pid status_a=0 status_b=0 merge_status=0
  local -a common=(--split "$split" --config "$config" --work-queue "$queue" --timestamp "$run_timestamp")
  (( resume )) && common+=(--resume)
  mkdir -p outputs "$results_root"
  echo "CoIN ${split}: GPUs ${gpu_a},${gpu_b}; config ${config}"
  switch_summary
  preflight || { echo 'Preflight failed.' >&2; exit 1; }
  total="$(preflight --count-only 2>/dev/null | tail -1)"
  if [[ -e "$queue" || -d "${results_root}/gpu${gpu_a}" || -d "${results_root}/gpu${gpu_b}" ]]; then
    (( resume )) || { echo 'Results already exist; choose a new timestamp or pass --resume' >&2; exit 1; }
    "$psgnav_python" tools/eval_schedule.py requeue --schedule "$queue"
  elif (( resume )); then
    echo "--resume requires ${queue}" >&2; exit 1
  else
    "$psgnav_python" tools/eval_schedule.py init --schedule "$queue" --total "$total" --max-episode-steps 500
  fi
  write_run_info "$total" dual
  echo "Logs: ${log_prefix}_gpu${gpu_a}.log ${log_prefix}_gpu${gpu_b}.log"
  echo "Results: ${results_root}/"
  exec 3>&1
  export PSGNAV_PROGRESS_FD=3
  started="$("$psgnav_python" -c 'import time; print(f"{time.time():.3f}")')"
  "$psgnav_python" -u tools/eval_schedule.py watch --schedule "$queue" --gpus "$gpu_list" --started "$started" &
  watch_pid=$!
  set +e
  pid_a=""; pid_b=""
  trap '[[ -n "${pid_a}" ]] && kill "${pid_a}" 2>/dev/null; [[ -n "${pid_b}" ]] && kill "${pid_b}" 2>/dev/null; true' INT TERM
  CUDA_VISIBLE_DEVICES="$gpu_a" bash "$script_path" --worker-gpu "$gpu_a" "${common[@]}" & pid_a=$!
  CUDA_VISIBLE_DEVICES="$gpu_b" bash "$script_path" --worker-gpu "$gpu_b" "${common[@]}" & pid_b=$!
  if ! wait -n "$pid_a" "$pid_b"; then
    echo 'A worker failed; stopping both workers. See the GPU logs.' >&2
    kill "$pid_a" "$pid_b" 2>/dev/null || true
  fi
  wait "$pid_a"; status_a=$?
  wait "$pid_b"; status_b=$?
  trap - INT TERM
  "$psgnav_python" tools/eval_schedule.py stop-watch --schedule "$queue"
  wait "$watch_pid"
  "$psgnav_python" tools/eval_schedule.py merge --results-root "$results_root" --gpus "$gpu_list" --total "$total"
  merge_status=$?
  set -e
  echo "gpu${gpu_a} exit ${status_a}; gpu${gpu_b} exit ${status_b}; merge exit ${merge_status}"
  summarize
  (( status_a == 0 && status_b == 0 && merge_status == 0 )) || { echo 'Evaluation incomplete.'; exit 1; }
}

run_single() {
  local log_file="${log_prefix}_gpu${single_gpu}.log" total status
  local -a cmd=("${nav_cmd[@]}" --episode_start 0)
  (( resume )) && cmd+=(--resume)
  mkdir -p outputs
  { echo "CoIN ${split}: GPU ${single_gpu}; config ${config}"; switch_summary
    echo "Log: ${log_file}"; echo "Results: ${results_root}/experiment_0/"; } | tee -a "$log_file"
  preflight >>"$log_file" 2>&1 || { echo "Preflight failed. Log: ${log_file}" >&2; exit 1; }
  total="$(preflight --count-only 2>/dev/null | tail -1)"
  cmd+=(--episode_end "$total")
  write_run_info "$total" single
  exec 3>&1
  set +e
  CUDA_VISIBLE_DEVICES="$single_gpu" PSGNAV_LOG_FILE="$log_file" PSGNAV_PROGRESS_FD=3 \
    "${cmd[@]}" >>"$log_file" 2>&1
  status=$?
  set -e
  summarize
  (( status == 0 )) || { echo "Evaluation interrupted (exit ${status}). Log: ${log_file}" >&2; exit "$status"; }
}

if [[ -n "$worker_gpu" ]]; then
  run_worker
elif [[ -n "$single_gpu" ]]; then
  run_single
else
  run_dual
fi
