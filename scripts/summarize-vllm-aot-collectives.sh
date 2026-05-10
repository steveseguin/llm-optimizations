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

printf 'cache=%s\n' "$cache"
printf 'all_reduce_comment_lines='
rg -n "Topologically Sorted Source Nodes: \\[.*all_reduce" "$cache" -g '*.py' | wc -l
printf 'all_reduce_call_lines='
rg -n "_c10d_functional\\.all_reduce_" "$cache" -g '*.py' | wc -l
printf 'wait_tensor_call_lines='
rg -n "_c10d_functional\\.wait_tensor" "$cache" -g '*.py' | wc -l
printf 'rms_int4_lines='
rg -n "vllm_ir\\.rms_norm.*_xpu_C\\.int4_gemm_w4a16|rms_norm_default, int4_gemm_w4a16" "$cache" -g '*.py' | wc -l
printf 'fused_add_rms_lines='
rg -n "fused_add_rms_norm|vllm_ir\\.fused_add_rms_norm" "$cache" -g '*.py' | wc -l

echo
echo "files_with_collectives:"
rg -l "_c10d_functional\\.all_reduce_" "$cache" -g '*.py' | sort

echo
echo "sample_collective_context:"
rg -n "Topologically Sorted Source Nodes: \\[all_reduce|_c10d_functional\\.all_reduce_|_c10d_functional\\.wait_tensor|rms_norm_default, int4_gemm_w4a16" "$cache" -g '*.py' | head -120
