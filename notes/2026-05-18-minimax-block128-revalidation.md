# MiniMax M2.7 block128 revalidation

Date: 2026-05-18

Scope: revalidate whether FlashAttention/default XPU attention with block size 128 can outperform the current strict-quality MiniMax M2.7 AutoRound TP4 baseline on 4x Intel Arc Pro B70.

## Baseline for comparison

Current promoted strict-quality baseline:

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Engine: `vllm 0.20.1-local`, XPU, TP4
- Hardware: 4x Intel Arc Pro B70 32GB
- Shape: prompt 512, output 1536, context 2048, batch 1
- Args: FlashAttention/default XPU attention, block size 256, `max_num_batched_tokens=512`, prefix cache off, async engine, PIECEWISE XPU graph
- Mean output tok/s: `70.00635289969598`
- Mean total tok/s: `93.34180386626133`
- LocalMaxxing id: `cmpahyaas002gmn01lk0625he`
- Data: `/home/steve/llm-optimizations-publish/data/minimax-m27-flash-piecewise-strict-revalidated-20260518.json`

## Candidate: block128, MBT1024

Command shape:

- `ATTENTION_BACKEND=default`
- `BLOCK_SIZE=128`
- `MAX_BATCHED_TOKENS=1024`
- `MAX_MODEL_LEN=2048`
- `INPUT_LEN=512`
- `OUTPUT_LEN=1536`
- `BENCH_REPEATS=3`
- strict quality enabled before benchmark
- compile config: `{"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE"}`

Result: rejected before benchmarking.

The first exact canary (`raw145-n64-exact`) failed with a combined token hash mismatch and degenerate output:

- Generated token count: 64
- Distinct generated token count: 1
- First distinct generated token: `0`
- NUL token count: 64
- Printable non-space chars: 0
- `control_char_output=true`
- `degenerate_output=true`
- Failed combined token hash: `3dc2cfd048d538e0707c7a833de67bbbd886730aa22a8f15b2a7f9ae01c6b15d`
- AOT hash observed for this failed candidate: `27004bb3d576a7e2112e9ac2c9878f13d8b0e9421a3e841d99ca978919f3c959`

Artifacts:

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-flash-block128-mbt1024-strict-revalidate-strict-tp4-ctx2048-mbt1024-bs128-20260518T031055Z-summary.json`
- Failed canary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-flash-block128-mbt1024-strict-revalidate-strict-tp4-ctx2048-mbt1024-bs128-20260518T031055Z-quality/raw145-n64-exact.json`

Decision: do not use `block_size=128` with `max_num_batched_tokens=1024` for this path. It can corrupt output before any meaningful speed test.

## Candidate: block128, MBT512

Command shape:

- `ATTENTION_BACKEND=default`
- `BLOCK_SIZE=128`
- `MAX_BATCHED_TOKENS=512`
- `MAX_MODEL_LEN=2048`
- `INPUT_LEN=512`
- `OUTPUT_LEN=1536`
- `BENCH_REPEATS=3`
- strict quality enabled before benchmark
- compile config: `{"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE"}`

Quality result: passed.

The candidate passed:

- `raw145-n64-exact`
- `raw145-n256-exact`
- semantic suite, two greedy repeats
- arithmetic repeat, 16 greedy repeats
- extended six-pack, two greedy repeats

AOT hash observed for this candidate: `96d74654910c9a809fed1fee1d8f2c66e21ba0578d069f4a16037abe3043b421`

Benchmark result after quality gate:

| Repeat | Output tok/s | Total tok/s |
| --- | ---: | ---: |
| 1 | 69.38338765613202 | 92.51118354150937 |
| 2 | 70.02855006226241 | 93.37140008301655 |
| 3 | 69.34202527861042 | 92.4560337048139 |
| Mean | 69.58465433233495 | 92.77953910977995 |

Artifacts:

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-flash-block128-mbt512-strict-revalidate-strict-tp4-ctx2048-mbt512-bs128-20260518T031639Z-summary.json`
- Quality dir: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-flash-block128-mbt512-strict-revalidate-strict-tp4-ctx2048-mbt512-bs128-20260518T031639Z-quality`

Decision: quality-safe but not faster than the promoted block256/MBT512 strict baseline. Do not promote and do not submit as a new LocalMaxxing result.

## Takeaway

Block128 is not the next optimization path for the current MiniMax M2.7 AutoRound TP4 recipe:

- MBT1024 is unsafe and produced NUL/control-token corruption.
- MBT512 is safe but averages `0.60%` below the current strict baseline.
- The promoted block256/MBT512 FlashAttention/default path remains the honest baseline.

Next work should focus on reducing or fusing the remaining graph-captured collectives in the block256 path, especially attention output projection and MoE hidden-state boundaries, rather than changing block size.
