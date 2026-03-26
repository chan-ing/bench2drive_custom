#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${1:-}"
NUM_TASKS="${2:-}"
if [[ -z "${NUM_GPUS}" || -z "${NUM_TASKS}" ]]; then
  echo "Usage: $0 <NUM_GPUS> <NUM_TASKS>"
  echo "Example: $0 4 8"
  exit 1
fi

IMAGE="jjanggu84/carla-0.9.15-maps-lm11:latest"

BASE_PORT=20200
BASE_TM_PORT=40200
STEP=200

CARLA_EXTRA_ARGS="-RenderOffScreen -graphicsadapter=0 -nosound"
EXPORT_DISPLAY_ENV=1   # 1: -e DISPLAY=$DISPLAY / 0: 안 넣기

pad2() { printf "%02d" "$1"; }

# 실행 전 컨테이너 정리: Carla_01 ~ Carla_<NUM_TASKS>
echo "[CLEAN] remove containers"
if [[ "${NUM_TASKS}" -eq 0 ]]; then
  echo "[CLEAN] NUM_TASKS=0 -> remove ALL containers whose name starts with Carla_"
  docker ps -a --format '{{.Names}}' | grep -E '^Carla_[0-9]+' | xargs -r docker rm -f
else
  echo "[CLEAN] removing Carla_01..Carla_$(pad2 "${NUM_TASKS}") if exist"
  for ((i=1; i<=NUM_TASKS; i++)); do
    docker rm -f "Carla_$(pad2 "${i}")" >/dev/null 2>&1 || true
  done
fi

start_one() {
  local task_id="$1"
  local gpu_id="$2"

  local rpc_port=$((BASE_PORT + task_id * STEP))
  local tm_port=$((BASE_TM_PORT + task_id * STEP))

  local idx=$((task_id + 1))
  local name="Carla_$(pad2 "${idx}")"

  local display_args=()
  if [[ "${EXPORT_DISPLAY_ENV}" -eq 1 ]]; then
    display_args=(-e "DISPLAY=${DISPLAY:-}")
  fi

  echo "[START] task=${task_id} -> GPU=${gpu_id}, name=${name}, rpc=${rpc_port}, tm=${tm_port}"

  docker run -d --name "${name}" \
    --gpus "\"device=${gpu_id}\"" \
    --net=host \
    "${display_args[@]}" \
    "${IMAGE}" \
    /bin/bash -lc \
    "./CarlaUE4.sh -carla-rpc-port=${rpc_port} --traffic-manager-port ${tm_port} ${CARLA_EXTRA_ARGS}" \
    >/dev/null
}

# 실행
for ((t=0; t<NUM_TASKS; t++)); do
  g=$((t % NUM_GPUS))
  start_one "${t}" "${g}"
done

echo "Launched ${NUM_TASKS} CARLA containers."
docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -E '^Carla_' || true