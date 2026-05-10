#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 /path/to/torch_aot_compile/<hash>/inductor_cache" >&2
  exit 2
fi

cache="$1"
if [ ! -d "$cache" ]; then
  echo "cache directory not found: $cache" >&2
  exit 1
fi

count_rg() {
  local pattern="$1"
  (rg -n "$pattern" "$cache" -g '*.py' 2>/dev/null || true) | wc -l
}

printf 'cache=%s\n' "$cache"
printf 'all_reduce_comment_lines='
count_rg "Topologically Sorted Source Nodes: \\[.*all_reduce"
printf 'all_reduce_call_lines='
count_rg "^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*: .* = torch\\.ops\\._c10d_functional\\.all_reduce"
printf 'wait_tensor_call_lines='
count_rg "^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*: .* = torch\\.ops\\._c10d_functional\\.wait_tensor"
printf 'rms_int4_lines='
count_rg "vllm_ir\\.rms_norm.*_xpu_C\\.int4_gemm_w4a16|rms_norm_default, int4_gemm_w4a16"
printf 'fused_add_rms_lines='
count_rg "fused_add_rms_norm|vllm_ir\\.fused_add_rms_norm"

echo
echo "files_with_collectives:"
rg -l "_c10d_functional\\.all_reduce" "$cache" -g '*.py' 2>/dev/null | sort || true

echo
echo "sample_collective_context:"
rg -n "Topologically Sorted Source Nodes: \\[all_reduce|^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*: .* = torch\\.ops\\._c10d_functional\\.all_reduce|^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*: .* = torch\\.ops\\._c10d_functional\\.wait_tensor|rms_norm_default, int4_gemm_w4a16" "$cache" -g '*.py' 2>/dev/null | head -120 || true
