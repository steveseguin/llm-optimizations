# MiniMax MoE Trace And Tile Sweep Negative

Date: 2026-05-18

## Baseline

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Promoted baseline: `81.758267` output tok/s, `109.011023` total tok/s
- Baseline recipe: logits-to-llm-scaler INT4 MoE work-sharing path, XPU FlashAttention v2, PIECEWISE graph, `MAX_BATCHED_TOKENS=512`

## Trace Finding

The first `LLM_SCALER_MOE_TRACE_KERNELS=1` attempt was not valid because `--mode eager` did not imply `enforce_eager=True`; vLLM still compiled graph ranges and the engine stalled after compile.

An enforced-eager rerun completed and passed the short raw canary:

- JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/moe-kernel-trace-current-best-enforce-eager-n4-20260518T111622Z.json`
- Log: `/home/steve/bench-results/minimax-m2.7-strict-candidates/moe-kernel-trace-current-best-enforce-eager-n4-20260518T111622Z.log`
- Result: passed, hash `2ecb10ff683aae9260eaee665272a325269e916ed4ab73cc22dafe42bd0ae82f`
- Valid stderr samples showed the expected work-sharing kernels:
  - `moe ws up routed cutlass int4`
  - `moe ws down cutlass int4`

The trace is diagnostic-only because it inserts waits and stderr writes. Multi-worker stderr interleaving means the aggregate numbers are approximate, but clean entries showed enough up-kernel wait/outlier signal to try an up-tile sweep.

## Tile Sweep

New opt-in source knobs were added to `moe_int4.sycl`:

- `VLLM_XPU_MOE_WS_UP_NTILE`: force routed up tile size to `2`, `4`, or `8`.
- `VLLM_XPU_MOE_WS_DOWN_HTILE`: force down tile size to `4` or `8`.

Build artifacts:

- Source SHA256: `7064e92719c598a12d0727bc71a9d134dac166f0a9b77ea6f04c06bf50039c3e`
- Extension SHA256: `5d6e85788590adc769a25b0c2606266fec68b92856a0384f1756a73ea261483c`
- Build log: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/build-moe-int4-u4-oneapi2025-20260518T112129Z.log`

## Outcomes

`VLLM_XPU_MOE_WS_UP_NTILE=4`:

- Raw n64 exact passed before full gate.
- Full strict quality gate passed: raw145 n64/n256 exact, semantic suite, 16-repeat arithmetic, extended sixpack.
- Benchmark mean: `79.236469` output tok/s, `105.648625` total tok/s.
- Decision: reject as slower than promoted `81.758267` / `109.011023`.
- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-ws-up-ntile4-strict-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T114044Z-summary.json`

`VLLM_XPU_MOE_WS_UP_NTILE=8`:

- The graph-mode raw canary compiled, then stalled in shared-memory broadcast wait after compile.
- No result JSON was produced; the run was terminated.
- Decision: reject as graph-unsafe/stalled.
- Log: `/home/steve/bench-results/minimax-m2.7-strict-candidates/ws-up-ntile8-raw145-n64-20260518T113156Z.log`

`VLLM_XPU_MOE_WS_DOWN_HTILE=8`:

- Structural corruption checks passed, but the exact raw hash changed.
- Observed hash: `21404821eb70a2ee3de9e82c039b5cbb5c9eef884c5019579f442c6a272a9c5a`
- Expected promoted raw n64 hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- Decision: reject without benchmark.
- JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/ws-down-htile8-raw145-n64-20260518T115947Z.json`

## Decision

Do not promote or submit any of these tile variants to LocalMaxxing. The useful learning is that simple tile-width changes do not improve the promoted logits-WS path:

- Wider up tile `4` is exact but slower.
- Wider up tile `8` is graph-unsafe in this runtime.
- Wider down tile `8` changes the exact greedy output.

Next work should avoid simple tile reties and instead target graph-safe MoE work sharing/epilogue structure, final logits/lm-head cost, or cleaner non-invasive timing.
