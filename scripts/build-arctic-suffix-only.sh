#!/usr/bin/env bash
set -euo pipefail

ARCTIC_VERSION="${ARCTIC_VERSION:-0.1.2}"
DEST="${DEST:-/home/steve/src/arctic-suffix-only}"
WORKDIR="${WORKDIR:-/mnt/fast-ai/tmp/arctic-suffix-only-build}"
VENV="${VENV:-/home/steve/.venvs/vllm-xpu}"

source "$VENV/bin/activate"

python -m pip install --no-deps "nanobind==2.9.2"

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
tarball="$WORKDIR/arctic_inference-$ARCTIC_VERSION.tar.gz"
if [ -n "${ARCTIC_TARBALL:-}" ]; then
  cp "$ARCTIC_TARBALL" "$tarball"
elif [ -f "/mnt/fast-ai/tmp/arctic-pip/arctic_inference-$ARCTIC_VERSION.tar.gz" ]; then
  cp "/mnt/fast-ai/tmp/arctic-pip/arctic_inference-$ARCTIC_VERSION.tar.gz" "$tarball"
else
  python - "$ARCTIC_VERSION" "$tarball" <<'PY'
import json
import sys
import urllib.request

version, out = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(
        f"https://pypi.org/pypi/arctic-inference/{version}/json") as r:
    meta = json.load(r)
sdists = [u for u in meta["urls"] if u["packagetype"] == "sdist"]
if not sdists:
    raise SystemExit(f"no sdist found for arctic-inference=={version}")
urllib.request.urlretrieve(sdists[0]["url"], out)
PY
fi

src="$WORKDIR/src"
mkdir -p "$src"
tar -xf "$tarball" -C "$src" --strip-components=1

rm -rf "$DEST"
mkdir -p "$DEST/arctic_inference/suffix_decoding" "$DEST/csrc/suffix_decoding"
cp "$src/arctic_inference/__init__.py" "$DEST/arctic_inference/__init__.py"
cp "$src/arctic_inference/suffix_decoding/__init__.py" "$DEST/arctic_inference/suffix_decoding/__init__.py"
cp "$src/arctic_inference/suffix_decoding/cache.py" "$DEST/arctic_inference/suffix_decoding/cache.py"
cp "$src/csrc/suffix_decoding/CMakeLists.txt" "$DEST/csrc/suffix_decoding/CMakeLists.txt"
cp "$src/csrc/suffix_decoding/bindings.cc" "$DEST/csrc/suffix_decoding/bindings.cc"
cp "$src/csrc/suffix_decoding/int32_map.h" "$DEST/csrc/suffix_decoding/int32_map.h"
cp "$src/csrc/suffix_decoding/suffix_tree.cc" "$DEST/csrc/suffix_decoding/suffix_tree.cc"
cp "$src/csrc/suffix_decoding/suffix_tree.h" "$DEST/csrc/suffix_decoding/suffix_tree.h"

cmake -S "$DEST/csrc/suffix_decoding" \
  -B "$DEST/build" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$DEST/arctic_inference/suffix_decoding"
cmake --build "$DEST/build" --target _C -j"$(nproc)"

PYTHONPATH="$DEST:${PYTHONPATH:-}" python - <<'PY'
from vllm.utils.import_utils import has_arctic_inference
from arctic_inference.suffix_decoding import SuffixDecodingCache

assert has_arctic_inference(), "vLLM does not see arctic_inference"
cache = SuffixDecodingCache(max_tree_depth=8, max_cached_requests=10)
cache.start_request("smoke", [1, 2, 3, 1, 2, 3, 4, 5])
cache.add_active_response("smoke", [1, 2, 3])
draft = cache.speculate("smoke", [1, 2, 3], max_spec_tokens=3, min_token_prob=0.0)
assert draft.token_ids, "suffix decoder returned no draft tokens"
print("arctic suffix-only smoke ok", draft.token_ids, draft.match_len)
PY

printf 'DEST=%s\n' "$DEST"
