#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
BASE_CKPT="${BASE_CKPT:-checkpoints/20260513_075638_head_only/best.pt}"
STAMP="$(date +%Y%m%d_%H%M)"
QUEUE_DIR="logs/${STAMP}_ablation_queue"
TIMEOUT_TRAIN="${TIMEOUT_TRAIN:-3h}"
TIMEOUT_E0="${TIMEOUT_E0:-2h}"
mkdir -p "${QUEUE_DIR}"

run_with_status() {
  local name="$1"
  local timeout_value="$2"
  local log_file="$3"
  shift 3
  set +e
  timeout "${timeout_value}" "$@" > "${log_file}" 2>&1
  local status="$?"
  set -e
  if [ "${status}" -eq 0 ]; then
    echo "SUCCESS" > "${log_file}.status"
  elif [ "${status}" -eq 124 ]; then
    echo "TIMEOUT" > "${log_file}.status"
  else
    echo "FAILED:${status}" > "${log_file}.status"
  fi
  echo "[$(date --iso-8601=seconds)] ${name} status=$(cat "${log_file}.status")" | tee -a "${QUEUE_DIR}/queue.log"
  return 0
}

run_threshold_search() {
  local name="${STAMP}_E0_threshold_search"
  local out_dir="analysis/${name}"
  local log_file="${QUEUE_DIR}/E0_threshold_search.log"
  echo "[$(date --iso-8601=seconds)] start E0 threshold search" | tee -a "${QUEUE_DIR}/queue.log"
  run_with_status "E0_threshold_search" "${TIMEOUT_E0}" "${log_file}" \
    env CUDA_VISIBLE_DEVICES="${GPU}" python -u threshold_search.py \
    --checkpoint "${BASE_CKPT}" \
    --dataset-root dataset \
    --split val \
    --device cuda \
    --output-dir "${out_dir}"
  echo "[$(date --iso-8601=seconds)] done E0 threshold search" | tee -a "${QUEUE_DIR}/queue.log"
}

run_train() {
  local config="$1"
  local method="$2"
  local run_name="${STAMP}_${method}"
  local log_dir="logs/${run_name}"
  local log_file="${log_dir}/train.log"
  mkdir -p "${log_dir}"
  {
    echo "run_name=${run_name}"
    echo "config=${config}"
    echo "gpu=${GPU}"
    echo "started_at=$(date --iso-8601=seconds)"
  } > "${log_dir}/run_info.txt"
  echo "[$(date --iso-8601=seconds)] start ${method}" | tee -a "${QUEUE_DIR}/queue.log"
  run_with_status "${method}" "${TIMEOUT_TRAIN}" "${log_file}" \
    env CUDA_VISIBLE_DEVICES="${GPU}" python -u train.py \
    --config "${config}" \
    --device cuda \
    --run-name "${run_name}"
  echo "[$(date --iso-8601=seconds)] done ${method}" | tee -a "${QUEUE_DIR}/queue.log"
}

run_threshold_search
run_train configs/E1_head_food28_chart18_other18.yaml E1_food28_chart18_other18
run_train configs/E2_head_food22_chart21_other21.yaml E2_food22_chart21_other21
run_train configs/E3_head_food_pos_weight_1_5.yaml E3_food_pos_weight_1_5
run_train configs/E4_head_light_food_aug.yaml E4_light_food_aug

echo "all done: ${QUEUE_DIR}"
