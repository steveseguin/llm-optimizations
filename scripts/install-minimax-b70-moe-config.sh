#!/usr/bin/env bash
set -euo pipefail

VENV="${VENV:-/home/steve/.venvs/vllm-xpu}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

src="$REPO_ROOT/configs/vllm/minimax-m27-b70-int4-w4a16-moe-hybrid-20260508.json"
dst_dir="$VENV/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/configs"
dst="$dst_dir/E=256,N=384,device_name=Intel(R)_Graphics_[0xe223],dtype=int4_w4a16.json"

install -d "$dst_dir"
install -m 0644 "$src" "$dst"
printf 'installed=%s\n' "$dst"
