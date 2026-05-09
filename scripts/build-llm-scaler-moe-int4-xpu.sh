#!/usr/bin/env bash
set -eo pipefail

SRC="${SRC:-/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm}"
VENV="${VENV:-/home/steve/.venvs/vllm-xpu}"
ONEAPI_COMPILER_ENV="${ONEAPI_COMPILER_ENV:-/opt/intel/oneapi/compiler/2025.3/env/vars.sh}"
OUTDIR="${OUTDIR:-/home/steve/bench-results/minimax-m2.7-autoround-vllm}"
MAX_JOBS="${MAX_JOBS:-2}"

mkdir -p "$OUTDIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
log="$OUTDIR/build-moe-int4-u4-oneapi2025-${ts}.log"

source "$ONEAPI_COMPILER_ENV" >/dev/null 2>&1
source "$VENV/bin/activate"

cd "$SRC"
rm -rf build python/custom_esimd_kernels_vllm/moe_int4_ops*.so
MAX_JOBS="$MAX_JOBS" TORCH_XPU_ARCH_LIST=bmg \
  python setup_moe_int4_only.py build_ext --inplace -v > "$log" 2>&1

printf 'log=%s\n' "$log"
