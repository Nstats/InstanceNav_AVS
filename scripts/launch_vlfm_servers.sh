#!/usr/bin/env bash
# Start the four VLFM model servers (GroundingDINO, BLIP-2 ITM, MobileSAM, YOLOv7) for
# NAV_BASE=vlfm, in a detached tmux session, and wait until all ports listen.
# Same commands as the official scripts/launch_vlm_servers.sh of VLFM, run from
# VLFM_ROOT with its vendored dependencies.
#
# Usage: bash scripts/launch_vlfm_servers.sh [--gpu N] [--session NAME]
#        bash scripts/launch_vlfm_servers.sh --stop [--session NAME]
set -euo pipefail

: "${VLFM_ROOT:=/data/lxj/vlfm}"
: "${VLFM_PYTHON:=${PSGNAV_PYTHON:-/home/temp/.conda/envs/psgnav-instance/bin/python}}"
: "${GROUNDING_DINO_PORT:=12181}"
: "${BLIP2ITM_PORT:=12182}"
: "${SAM_PORT:=12183}"
: "${YOLOV7_PORT:=12184}"
gpu="${VLFM_SERVER_GPU:-0}"
session="vlfm_servers"
stop=0
while (( $# )); do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --session) session="$2"; shift 2 ;;
    --stop) stop=1; shift ;;
    -h|--help) sed -n 2,8p "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if (( stop )); then
  tmux kill-session -t "${session}" 2>/dev/null && echo "Stopped tmux session ${session}" || echo "No session ${session}"
  exit 0
fi

listening() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
ports=("${GROUNDING_DINO_PORT}" "${BLIP2ITM_PORT}" "${SAM_PORT}" "${YOLOV7_PORT}")
all_up=1
for port in "${ports[@]}"; do listening "${port}" || all_up=0; done
if (( all_up )); then
  echo "VLFM servers already listening on ${ports[*]}"
  exit 0
fi
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session ${session} exists but not every port listens; stop it first (--stop)" >&2
  exit 1
fi

env_prefix="cd '${VLFM_ROOT}' && export CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH='${VLFM_ROOT}/.vlfm-deps:${VLFM_ROOT}'"
# The official YOLOv7 checkpoint pickles its model class (torch >= 2.6 refuses by default).
env_prefix+=" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MOBILE_SAM_CHECKPOINT=data/mobile_sam.pt"
env_prefix+=" GROUNDING_DINO_CONFIG=GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
env_prefix+=" GROUNDING_DINO_WEIGHTS=data/groundingdino_swint_ogc.pth CLASSES_PATH=vlfm/vlm/classes.txt"
tmux new-session -d -s "${session}" -n gdino
tmux send-keys -t "${session}:gdino" "${env_prefix} && '${VLFM_PYTHON}' -m vlfm.vlm.grounding_dino --port ${GROUNDING_DINO_PORT}" C-m
tmux new-window -t "${session}" -n blip2itm
tmux send-keys -t "${session}:blip2itm" "${env_prefix} && '${VLFM_PYTHON}' -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT}" C-m
tmux new-window -t "${session}" -n sam
tmux send-keys -t "${session}:sam" "${env_prefix} && '${VLFM_PYTHON}' -m vlfm.vlm.sam --port ${SAM_PORT}" C-m
tmux new-window -t "${session}" -n yolov7
tmux send-keys -t "${session}:yolov7" "${env_prefix} && '${VLFM_PYTHON}' -m vlfm.vlm.yolov7 --port ${YOLOV7_PORT}" C-m

echo "Started tmux session ${session} on GPU ${gpu}; waiting for ports ${ports[*]} (up to 300 s)"
for (( waited = 0; waited < 300; waited += 5 )); do
  all_up=1
  for port in "${ports[@]}"; do listening "${port}" || all_up=0; done
  if (( all_up )); then
    echo "VLFM servers ready after ${waited} s (tmux attach -t ${session})"
    exit 0
  fi
  sleep 5
done
echo "Timed out; inspect with: tmux attach -t ${session}" >&2
exit 1
