# MiniMax Runtime-Guarded Baseline Refresh

Date: 2026-05-17

## Result

After the gather/broadcast erratum, I added a runtime import guard and re-ran the quality-promoted MiniMax M2.7 AutoRound path.

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU, TP4
- Quantization: AutoRound INT4 W4A16
- Shape: p512 / n1536 / batch 1 / context 2048
- Mean output throughput: `61.317497` tok/s
- Mean total throughput: `81.756663` tok/s
- Repeats: `60.894366`, `61.740629` output tok/s
- LocalMaxxing: `cmp9q9fzn04cto401tjcila06`

This supersedes the earlier mislabeled gather/broadcast row for interpretation. The older `cmp940h1703tpo401scj5tftf` result remains a valid stricter extended-gate reference; this refresh adds explicit runtime import verification and a slightly higher two-run mean.

## Guardrails Added

The new `scripts/inspect-vllm-runtime.py` imports the active vLLM modules from the running Python environment and records:

- module paths
- sha256 and size
- import stdout/stderr
- required/forbidden marker checks

Both MiniMax benchmark wrappers now call it before loading the model. For this run, the active logits processor was:

`/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/vllm/model_executor/layers/logits_processor.py`

sha256:

`c032b08de08e929fb38a54624401eeae83c618dd19edfb97e5ecdec379fbe254`

The guard required `logits.local_argmax_pair_all_gather`, so the run could not silently use an unpatched source-tree file.

The strict quality wrapper also now has a startup watchdog. If vLLM/oneCCL does not reach a model-load milestone within `QUALITY_STARTUP_GUARD_SECONDS`, it logs the tail and terminates the candidate rather than burning the full quality timeout.

## Quality Gates

All quality gates passed before throughput was measured:

- raw145 n64 exact combined token hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact combined token hash: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic repeat suite: PASS prompt, arithmetic `42`, and `def add_one` / `return x + 1`
- arithmetic repeat suite: 8 greedy repeats, all deterministic, all matched `42`

## Launch Summary

Important environment:

```bash
FI_TCP_IFACE=wlxe865d47e3a48
CCL_KVS_IFACE=wlxe865d47e3a48
USE_LLM_SCALER_MOE=1
VLLM_XPU_USE_LLM_SCALER_MOE=1
VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS=1
VLLM_XPU_ENABLE_XPU_GRAPH=1
VLLM_XPU_FORCE_GRAPH_WITH_COMM=1
VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1
VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1
VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1
VLLM_XPU_LOCAL_ARGMAX_DECODE=1
VLLM_XPU_LOCAL_ARGMAX_ASSUME_SAFE=1
VLLM_BENCH_TEMPERATURE=0
VLLM_RUNTIME_REQUIRE_MARKERS=logits.local_argmax_pair_all_gather
```

Benchmark command shape:

```bash
vllm bench throughput \
  --backend vllm \
  --model /mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround \
  --tokenizer /mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround \
  --trust-remote-code \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --distributed-executor-backend mp \
  --max-model-len 2048 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 1 \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 1536 \
  --random-range-ratio 0 \
  --num-prompts 1 \
  --disable-log-stats \
  --async-engine \
  --block-size 256 \
  --no-enable-prefix-caching \
  --attention-backend TRITON_ATTN \
  --compilation-config '{"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE"}'
```

## Artifacts

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimaxlogits-localargmax-pairgather-runtimeguard-repro2-strict-tp4-ctx2048-mbt512-bs256-20260517T114411Z-summary.json`
- Benchmark JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T115655Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T115948Z.json`
- LocalMaxxing payload: `data/localmaxxing-minimax-m27-autoround-runtimeguard-repro-p512n1536-20260517.payload.json`
- LocalMaxxing response: `data/localmaxxing-responses/minimax-m27-autoround-runtimeguard-repro-p512n1536-20260517.response.json`

## Next Leads

- Keep the pair-all-gather local-argmax path as the honest baseline.
- Do not promote gather/broadcast, packed allreduce, or two-allreduce variants unless they pass this same runtime guard and quality gate.
- Next speed work should target GPU-resident token handoff without new graph-time collectives, and then revisit EP/MoE placement only under the same quality gate.
