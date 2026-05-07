# Intel Arc Pro B70 LLM Lab Notes

Date: 2026-05-04

This note is the current public/reproducible summary for the B70 optimization work. The active technical plan is `plans/q4_0-gguf-b70-optimization-plan.md`; submitted benchmark IDs and exact payloads are recorded in `notes/localmaxxing-submissions-2026-05-04.md` and `data/localmaxxing-payloads-20260504.json.gz.b64`.

## Hardware And Constraints

- Host: Ubuntu 24.04.4 LTS, kernel 6.17.0-22-generic during the latest runs.
- CPU: AMD EPYC 9015.
- GPUs: 4x Intel Arc Pro B70 32GB, exposed through Level Zero selectors `0-3`.
- Power-limit / overclocking changes are intentionally out of scope. The current work is software-only.
- Single-card tests should isolate one device with `ONEAPI_DEVICE_SELECTOR=level_zero:N`.
- llama.cpp multi-GPU device syntax is slash-separated, for example `-dev SYCL0/SYCL1/SYCL2`.

## Current Best Results

Quality-preserving Q4_0 GGUF path, llama.cpp/SYCL:

- 1x B70 baseline: about `24.7 tok/s` decode.
- 2x B70 async-copy tensor split: `37.690 tok/s`, LocalMaxxing `cmoqkcqpv0006la04l5mtlj2q`.
- 2x B70 single-kernel allreduce: `39.849 tok/s`, LocalMaxxing `cmoqp6jpq0004lb04241n9ns3`.
- 2x B70 Q8 activation-cache validation, 512 prompt / 512 output: `40.487 tok/s`, LocalMaxxing `cmormylxz000fib04wodwo1ng`.
- 3x B70 single-kernel allreduce, selector/root order `2,1,3`: `41.737 tok/s`, LocalMaxxing `cmoqqed6s0007jv049wnizwle`.
- 3x B70 Q8 activation-cache short run, 512 prompt / 256 output: `42.432 tok/s`, LocalMaxxing `cmordq9t5000dl404x309pj48`.
- 3x B70 Q8 activation-cache validation, 512 prompt / 512 output: `41.659 tok/s`, LocalMaxxing `cmorn71e2000kib0415vo51vj`.
- 3x B70 current quality-cleared no-root Q4_0 stack with experimental flat fused Qwen35 `ssm_ba` GGUF, 512 prompt / 512 output: `50.130 tok/s`, LocalMaxxing `cmov6p4r7007tqr01yi8ug4un`.
- 3x B70 root-residual performance ceiling with `--poll 25`, 512 prompt / 512 output: `50.922 tok/s`, LocalMaxxing `cmouxjqao000npn01hxqn68td`, now marked suspect because later token/logit probing found the root-residual plus meta allreduce-add pair can diverge.
- 3x B70 final-rebuild root-residual rerun with flat fused `ssm_ba` GGUF: `50.687 tok/s` decode and `80.879 tok/s` total. Default-prompt root checks passed, but a two-token prompt follow-up timed out, so this is documented but not submitted/promoted.
- 4x B70 Q4_0 remains negative-scaling: `31.482 tok/s` without Q8 cache and `31.913 tok/s` with Q8 cache; LocalMaxxing `cmor2e5r00004jl04o99d26p8` and `cmornec37000okw040zl9563z`.

FP8 path, vLLM/XPU:

- Official `Qwen/Qwen3.6-27B-FP8` runs, but current XPU block-FP8/requant path is slow: TP2 512-output upper-bound `20.106 tok/s`, LocalMaxxing `cmorb75xb001ckz0489eqc9se`.
- Static `vrfai/Qwen3.6-27B-FP8` with patched XPU FlashAttention2 reaches `41.503 tok/s` on TP4 for 512 prompt / 512 output, LocalMaxxing `cmork3n3k000ujo04y73lbr1j`.
- TP4 also fits Qwen3.6 full configured context (`262144`) and reports `1,206,355` GPU KV-cache tokens.
- PP2 x TP2 is valid as a capacity fallback but slower for one sequence: `22.721 tok/s`, LocalMaxxing `cmormmlz0000bky04wpu4oc01`.
- FP8 KV cache is not a speed path and is quality-risky without proper scaling: `28.036 tok/s`, LocalMaxxing `cmornlh8g000vkw04yb57ukvl`.

INT4 AutoRound path:

- `Lorbus/Qwen3.6-27B-int4-AutoRound` produced strong vLLM/XPU speed results, including `45.2 tok/s` on 1x B70 and `49.1 tok/s` on 2x B70.
- These results are recorded on LocalMaxxing but are not counted as Q4_0 GGUF success because the quantization changes model fidelity.

## Important Implementation Artifacts

llama.cpp Q4_0/SYCL work:

- Combined diff: `patches/llama-cpp-db44417-b70-sycl-combined.diff.gz.b64`.
- Decode/apply guide: `patches/llama-cpp-db44417-b70-sycl-combined-diff.md`.
- Key runtime flags for the best Q4_0 runs:
  - `GGML_SYCL_ASYNC_CPY_TENSOR=1`
  - `GGML_SYCL_COMM_ALLREDUCE=1`
  - `GGML_SYCL_COMM_SINGLE_KERNEL=1`
  - `GGML_SYCL_Q8_CACHE=1`
- Benchmark harnesses:
  - `scripts/bench-qwen36-q4_0-gguf-sycl-matrix.sh`
  - `scripts/bench-qwen36-q4_0-gguf-vulkan-matrix.sh`

vLLM/XPU FP8 work:

- XPU FA2 singleton scale patch: `patches/vllm-xpu-fa2-compressed-tensors-scalar-scales.patch`.
- Qwen3.5/Qwen3.6 language-only vision skip patch: `patches/vllm-qwen35-language-model-only-skip-vision.patch`.
- FP8 result notes: `notes/2026-05-04-qwen36-fp8-b70-fa2.md` and `notes/2026-05-04-qwen36-fp8-full-context-topologies.md`.
- FP8 topology data: `data/qwen36-fp8-b70-topology-screens-20260504.json`.

## Current Diagnosis

- Single-card Q4_0 is not limited by flash attention, ubatch, graph capture, oneDNN, AOT alone, or a missing reordered MMVQ path.
- Reordered Q4_0 MMVQ is required; disabling it drops single-card speed to about `15 tok/s`.
- Multi-card Q4_0 improves through async tensor copies, direct allreduce, Q8 activation caching, graph fusions, fused small projections, and safe allreduce+ADD scheduling. Root-residual fused allreduce+ADD is a promising performance ceiling but is not quality-cleared until its ordering hazard with meta allreduce-add is fixed. Four-card scaling still regresses because each token pays many small 20 KiB reductions and narrower row shards lose matvec efficiency.
- Timing hooks show synchronized allreduce cost rises sharply with GPU count: roughly `1.718 ms/token` on 2 GPUs, `5.732 ms/token` on 3 GPUs, and `10.605 ms/token` on 4 GPUs for the same reduction pattern.
- Four-GPU root/order/topology sweeps are not enough; the next useful work is reducing the number of reductions or fusing delayed reductions through safe graph regions.
- MiniMax M2.7 currently proves capacity pressure and split-expert allocator issues, not useful performance: tensor split is unsupported for `minimax-m2`, layer split fails large SYCL allocations, and row split currently fails on GPU expert tensor allocation unless experts are left CPU/file-backed.

## Current Next Steps

1. Keep Q4_0 single-card profiling focused on reordered MMVQ and activation quantization launch overhead.
2. Prototype fewer/fused Meta allreduces for Q4_0 multi-GPU; do not spend more time on simple root-copy or pairwise allreduce topology variants.
3. Use 3x B70 Q4_0 no-root fused beta/alpha at `50.130 tok/s` as the current quality-cleared speed point; treat root-residual `50.922 tok/s` as a performance ceiling until the correctness hazard is fixed.
4. Use static FP8 TP4 with patched XPU FA2 as the best high-fidelity four-card path for now.
5. Keep PP2 x TP2 as a capacity fallback for larger models, not a speed path for Qwen3.6 27B.
6. Mine Intel `llm-scaler` for reduce-scatter/all-gather, fused output-kernel, Gated DeltaNet, and speculative/MTP ideas, but do not assume it will run directly on Arc/B70.
