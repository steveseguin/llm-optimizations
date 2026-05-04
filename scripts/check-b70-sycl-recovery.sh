#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf}"
LLAMA_BENCH="${LLAMA_BENCH:-/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31-aot-dnn/bin/llama-bench}"

if [[ "${SOURCE_ONEAPI:-1}" == "1" && -f /opt/intel/oneapi/setvars.sh ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/setvars.sh --force >/dev/null
  set -u
fi

echo "== user/groups =="
id
if ! id -nG | tr ' ' '\n' | grep -qx render; then
  echo "missing render group for current shell"
  echo "fix: log out/in after usermod -aG render steve, or start a shell with newgrp render"
fi

echo
echo "== dri nodes =="
ls -l /dev/dri || true

echo
echo "== b70 pci devices =="
for devpath in /sys/bus/pci/devices/*; do
  [[ -f "$devpath/vendor" && -f "$devpath/device" ]] || continue
  vendor="$(<"$devpath/vendor")"
  device="$(<"$devpath/device")"
  [[ "$vendor" == "0x8086" && "$device" == "0xe223" ]] || continue
  printf '%s %s %s driver=' "$(basename "$devpath")" "$vendor" "$device"
  readlink "$devpath/driver" 2>/dev/null || printf 'none'
  printf '\n'
done

echo
echo "== stuck gpu probes =="
ps -eo pid,ppid,stat,comm,args | grep -E 'sycl-ls|llama-bench|xe/(un)?bind|pkill' | grep -v grep || true

echo
echo "== sycl-ls =="
timeout "${SYCL_LS_TIMEOUT:-20s}" sycl-ls

if [[ "${1:-}" == "--smoke" ]]; then
  echo
  echo "== llama-bench smoke =="
  # Keep the single-card smoke to one visible Level Zero GPU. With four B70s
  # exposed, single-card tests should not use level_zero:*.
  export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"
  export ZES_ENABLE_SYSMAN="${ZES_ENABLE_SYSMAN:-1}"
  export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS="${UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS:-1}"
  export GGML_SYCL_DISABLE_GRAPH="${GGML_SYCL_DISABLE_GRAPH:-0}"
  export GGML_SYCL_DISABLE_OPT="${GGML_SYCL_DISABLE_OPT:-0}"
  export GGML_SYCL_DISABLE_DNN="${GGML_SYCL_DISABLE_DNN:-0}"
  "$LLAMA_BENCH" \
    -m "$MODEL" \
    -dev "${DEVICE:-SYCL0}" \
    -ngl 99 \
    -p 0 \
    -n "${OUTPUT_TOKENS:-16}" \
    -sm none \
    -b 512 \
    -ub 64 \
    -ctk f16 \
    -ctv f16 \
    -t 8 \
    -fa 0 \
    -r 1 \
    -o jsonl
fi
