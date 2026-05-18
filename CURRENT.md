# Current Promoted Results

Date: 2026-05-18

## MiniMax M2.7

Current strict quality-passed speed result:

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Recipe: FP16 activations, AutoRound INT4 W4A16, default XPU FlashAttention v2, XPU PIECEWISE graph, exact MiniMax router-logits path feeding llm-scaler INT4 MoE work-sharing decode with `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`, `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`, and `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0`
- Shape: p512/n1536, ctx2048, batch 1
- Result: `82.404268` output tok/s, `109.872357` total tok/s, mean of four clean long repeats
- Quality: raw145 exact n64/n256 hashes, semantic suite, 16-repeat arithmetic, and extended sixpack all passed before benchmarking
- Delta: `+0.79%` output tok/s over the previous strict logits-WS promoted result and `+2.24%` over the earlier MoE-WS FlashAttention/PIECEWISE baseline
- LocalMaxxing: `cmpbifcx3013bmn01747cxix8`

Primary artifacts:

- `notes/2026-05-18-minimax-logits-ws-no-attn-delay-small-win.md`
- `data/minimax-m27-logits-ws-no-attn-delay-small-win-20260518.json`
- `data/localmaxxing-minimax-m27-autoround-logits-ws-no-attn-delay-p512n1536-20260518.payload.json`
- `data/localmaxxing-responses/minimax-m27-autoround-logits-ws-no-attn-delay-p512n1536-20260518.response.json`
- `notes/2026-05-18-minimax-logits-ws-strict-win.md`
- `data/minimax-m27-logits-ws-strict-win-20260518.json`
- `data/localmaxxing-minimax-m27-autoround-logits-ws-strict-p512n1536-20260518.payload.json`
- `data/localmaxxing-responses/minimax-m27-autoround-logits-ws-strict-p512n1536-20260518.response.json`
- `patches/minimax-logits-ws-path-20260518.md`

Previous promoted logits-WS baseline:

- Same exact router-logits-to-work-sharing path with delayed attention allreduce enabled: `81.758267` output tok/s and `109.011023` total tok/s, mean of two strict-gated repeats.
- Confirmation repeat: `81.197954` output tok/s, `108.263938` total tok/s; three-run mean was `81.571496` output tok/s.
- LocalMaxxing: `cmpay7th600bbmn01v6csyaro`
- Artifacts: `notes/2026-05-18-minimax-logits-ws-strict-win.md`, `data/minimax-m27-logits-ws-strict-win-20260518.json`

Previous MoE-WS baseline:

- `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1` without logits-to-WS routing: `80.602755` output tok/s and `107.470340` total tok/s.
- LocalMaxxing: `cmpasdq5v007nmn019elaut3s`
- Artifacts: `notes/2026-05-18-minimax-moe-ws-flash-piecewise-strict-win.md`, `data/minimax-m27-moe-ws-flash-piecewise-strict-win-20260518.json`

Recent MBT boundary follow-up:

- `MAX_BATCHED_TOKENS=768`: strict quality passed, `80.876005` output tok/s and `107.834674` total tok/s mean, only +0.34% output over MBT512.
- `MAX_BATCHED_TOKENS=832`: strict quality passed, but slower at `77.795833` output tok/s and `103.727777` total tok/s mean.
- `MAX_BATCHED_TOKENS=896`: unsafe; raw exact canaries passed but semantic repeat produced NUL/control-token corruption.
- `MAX_BATCHED_TOKENS=1024`: unsafe; raw145 n64 exact canary failed immediately with NUL/control-token corruption.
- Decision: keep MBT512 as the promoted public setting. No LocalMaxxing submission was made for the boundary sweep because no candidate gave a material quality-safe improvement.
- Artifacts: `notes/2026-05-18-minimax-mbt-boundary.md`, `data/minimax-m27-mbt-boundary-20260518.json`

Recent MoE-delay follow-up:

- `VLLM_MINIMAX_MOE_DELAY_ALLREDUCE=1` with the same MBT512 work-sharing FlashAttention/PIECEWISE recipe passed the full strict quality gate.
- Result: `79.481453` output tok/s and `105.975271` total tok/s mean, slower than the promoted baseline.
- Decision: do not promote and do not submit to LocalMaxxing. Keep effort on decode-critical collective and epilogue work.
- Artifacts: `notes/2026-05-18-minimax-moe-delay-negative.md`, `data/minimax-m27-moe-delay-negative-20260518.json`

Recent no-clone/final-hidden-clone follow-up:

- `VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1` plus `VLLM_MINIMAX_M2_CLONE_FINAL_HIDDEN=1` with the same MBT512 work-sharing FlashAttention/PIECEWISE recipe passed the full strict quality gate.
- Result: `80.791520` output tok/s and `107.722027` total tok/s mean.
- Decision: validated tie, not a material new win. Do not submit to LocalMaxxing because the `+0.23%` output delta over the promoted `80.602755` result is within run variance.
- Artifacts: `notes/2026-05-18-minimax-no-clone-clonefinal-retie.md`, `data/minimax-m27-no-clone-clonefinal-retie-20260518.json`

Recent logits-WS no-clone/final-hidden-clone follow-up:

- `VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1` plus `VLLM_MINIMAX_M2_CLONE_FINAL_HIDDEN=1` was retested on top of the current logits-to-work-sharing baseline.
- Result: strict quality passed, but mean speed was `81.021124` output tok/s and `108.028166` total tok/s, below the promoted `81.758267` / `109.011023` logits-WS baseline.
- Decision: not promoted and not submitted to LocalMaxxing. This makes further flag reties less interesting than measured timing around residual allreduce and final logits boundaries.
- Artifacts: `notes/2026-05-18-minimax-logits-ws-noclone-clonefinal-negative.md`, `data/minimax-m27-logits-ws-noclone-clonefinal-negative-20260518.json`

Recent decode-boundary timing:

- Synchronized diagnostics found final logits at about `0.86 ms/token`, with local lm-head projection larger than TP logits gathering.
- Eager per-layer labels identified three similar steady decode collectives: Q/K variance allreduce, attention delayed residual allreduce, and MoE expert output allreduce.
- Model-forward timing wrappers were not neutral in compiled graph and were reverted. Active `minimax_m2.py` and `logits_processor.py` hashes match the promoted runtime again.
- Artifacts: `notes/2026-05-18-minimax-decode-boundary-timing.md`, `data/minimax-m27-decode-boundary-timing-20260518.json`

Recent candidate-router repair follow-up:

- Top-16 candidate-router repair failed the first raw145 n64 exact token-hash gate, so it was rejected without benchmarking.
- Top-32 candidate-router repair passed the full strict quality gate, including raw145 n64/n256 exact hashes, semantic suite, 16-repeat arithmetic, and extended sixpack.
- Result: `80.008471` output tok/s and `106.677962` total tok/s mean, slower than the promoted `81.758267` / `109.011023` logits-WS baseline.
- Decision: do not promote and do not submit to LocalMaxxing. The candidate-repair path is quality-preserving at top32 but not faster than the exact router-logits WS path.
- Artifacts: `notes/2026-05-18-minimax-candidate-router-top32-negative.md`, `data/minimax-m27-candidate-router-top32-negative-20260518.json`

Recent WS internal scratch-reuse rejection:

- `VLLM_XPU_MINIMAX_WS_REUSE_INTERNAL=1` attempted to reuse internal top-k and intermediate tensors inside the exact MiniMax logits-WS C++ op.
- It failed the first raw145 n64 exact canary with NUL/control-token corruption, so it was rejected without benchmarking.
- A default-path raw145 n64 canary with the rebuilt extension and the reuse env unset passed the expected token hash, confirming the promoted runtime remains intact.
- Decision: do not enable static internal scratch reuse under XPU graph capture/replay. Future allocation-overhead work needs graph-safe lifetime management.
- Artifacts: `notes/2026-05-18-minimax-ws-internal-reuse-reject.md`, `data/minimax-m27-ws-internal-reuse-reject-20260518.json`

Recent MoE trace and tile-sweep follow-up:

- Enforced-eager `LLM_SCALER_MOE_TRACE_KERNELS=1` completed and showed the promoted path using the expected `moe ws up routed cutlass int4` and `moe ws down cutlass int4` kernels. Trace output is diagnostic-only because it inserts waits and multi-worker stderr interleaves.
- `VLLM_XPU_MOE_WS_UP_NTILE=4` passed the full strict quality gate but was slower at `79.236469` output tok/s and `105.648625` total tok/s.
- `VLLM_XPU_MOE_WS_UP_NTILE=8` stalled after graph compile; no result JSON.
- `VLLM_XPU_MOE_WS_DOWN_HTILE=8` failed exact raw-hash equivalence, producing `21404821eb70a2ee3de9e82c039b5cbb5c9eef884c5019579f442c6a272a9c5a` instead of the promoted `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`.
- Decision: do not promote and do not submit to LocalMaxxing. Simple WS tile reties are not the next useful path.
- Artifacts: `notes/2026-05-18-minimax-moe-trace-and-tile-negative.md`, `data/minimax-m27-moe-trace-and-tile-negative-20260518.json`

Recent greedy sampler fp32-skip follow-up:

- `VLLM_XPU_GREEDY_SKIP_LOGITS_FP32=1` skipped the sampler-side XPU logits-to-fp32 conversion for guarded greedy/no-logprobs/no-processor requests.
- It passed raw145 n64/n256 exact hashes, semantic suite, 16-repeat arithmetic, and extended sixpack.
- Result: `81.549421` output tok/s and `108.732562` total tok/s mean, slightly below the promoted `81.758267` / `109.011023` logits-WS baseline.
- Decision: do not promote and do not submit to LocalMaxxing. The sampler fp32 conversion is not the current bottleneck; the active runtime sampler was restored to the promoted behavior.
- Artifacts: `notes/2026-05-18-minimax-greedy-skip-logits-fp32-negative.md`, `data/minimax-m27-greedy-skip-logits-fp32-negative-20260518.json`, `patches/minimax-greedy-skip-logits-fp32-negative-20260518.md`

Recent WS top-k reuse rejection:

- `VLLM_XPU_MINIMAX_WS_REUSE_TOPK_ONLY=1` attempted a narrower graph scratch reuse than the earlier internal-reuse failure by reusing only MiniMax WS top-k tensors.
- It failed raw145 n64 exact immediately with NUL/control-token corruption: observed hash `242152df6909e5e25433f43875de5e51c210d146a22279611852b695bcf7d978` instead of `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`.
- The patch was reverted and a default raw145 n64 canary passed with the expected hash and no NUL/control output.
- Decision: do not use static thread-local top-k reuse under XPU graph replay. Future allocation work needs graph-owned buffers or explicit graph lifetime management.
- Artifacts: `notes/2026-05-18-minimax-ws-topk-reuse-reject.md`, `data/minimax-m27-ws-topk-reuse-reject-20260518.json`, `patches/minimax-ws-topk-reuse-rejected-20260518.md`

Recent safe hidden-state selection follow-up:

- `VLLM_XPU_SAFE_SAMPLE_HIDDEN_SELECT=1` used the existing guarded XPU path that returns the full hidden batch when sampled logits rows already cover the full batch, otherwise using `torch.index_select` instead of advanced indexing.
- It passed raw145 n64/n256 exact hashes, semantic suite, 16-repeat arithmetic, and extended sixpack under the fair default XPU FlashAttention v2 backend.
- Fair result: `81.914167` output tok/s and `109.218890` total tok/s mean, a `+0.19%` output delta versus the promoted `81.758267` / `109.011023` logits-WS baseline.
- Four-repeat confirmation result: strict quality passed again, but the mean was `81.379492` output tok/s and `108.505990` total tok/s, `-0.46%` versus the promoted logits-WS baseline.
- Decision: neutral/tie, not promoted and not submitted to LocalMaxxing. The delta is inside normal run variance.
- Erratum: the first `77.314354` output tok/s safe-selector run used the strict runner's older `TRITON_ATTN` default, so it is quality-valid diagnostic data but not a fair comparison against the promoted FlashAttention baseline.
- Artifacts: `notes/2026-05-18-minimax-safe-sample-hidden-select-negative.md`, `data/minimax-m27-safe-sample-hidden-select-negative-20260518.json`, `notes/2026-05-18-minimax-safe-hidden-repeatability.md`, `data/minimax-m27-safe-hidden-repeatability-20260518.json`, `patches/minimax-safe-sample-hidden-select-negative-20260518.md`

Recent logits-WS local-argmax follow-up:

- `VLLM_XPU_LOCAL_ARGMAX_DECODE=1` plus `VLLM_XPU_LOCAL_ARGMAX_ASSUME_SAFE=1` was retested on top of the current exact logits-to-work-sharing MiniMax baseline under the fair default XPU FlashAttention v2 backend.
- It passed raw145 n64/n256 exact hashes, semantic suite, 16-repeat arithmetic, and extended sixpack.
- Result: `72.980385` output tok/s and `97.307181` total tok/s mean, well below the promoted `81.758267` / `109.011023` logits-WS baseline.
- Decision: do not promote and do not submit to LocalMaxxing. Local argmax is quality-safe for this guarded greedy benchmark, but after the logits-WS MoE improvement the full-vocab logits/sampler path is not the current bottleneck.
- Artifacts: `notes/2026-05-18-minimax-logits-ws-localargmax-negative.md`, `data/minimax-m27-logits-ws-localargmax-negative-20260518.json`

Recent logits-WS Q/K RMS helper follow-up:

- `VLLM_MINIMAX_QK_RMS_XPU_HELPER=1` was retested on top of the current exact logits-to-work-sharing MiniMax baseline under the fair default XPU FlashAttention v2 backend.
- It passed raw145 n64/n256 exact hashes, semantic suite, 16-repeat arithmetic, and extended sixpack.
- Result: `81.441928` output tok/s and `108.589237` total tok/s mean, slightly below the promoted `81.758267` / `109.011023` logits-WS baseline.
- Decision: do not promote and do not submit to LocalMaxxing. The helper is quality-safe, but replacing only the local Q/K RMS math does not remove the decode-critical Q/K variance collective boundary.
- Artifacts: `notes/2026-05-18-minimax-logits-ws-qk-rms-helper-negative.md`, `data/minimax-m27-logits-ws-qk-rms-helper-negative-20260518.json`

Recent logits-WS Q/K RMS helper in-place allreduce follow-up:

- `VLLM_MINIMAX_QK_RMS_XPU_HELPER=1` plus `VLLM_MINIMAX_QK_RMS_XPU_HELPER_INPLACE_ALLREDUCE=1` was tested on top of the current no-attention-delay logits-WS baseline.
- It passed raw145 n64/n256 exact hashes, semantic suite, 16-repeat arithmetic, and extended sixpack.
- Result: `81.939211` output tok/s and `109.252281` total tok/s mean, below the current `82.404268` / `109.872357` no-attention-delay baseline.
- Decision: do not promote and do not submit to LocalMaxxing. The localized in-place Q/K variance allreduce is quality-safe, but speed is worse; future Q/K work needs boundary fusion rather than wrapper substitution.
- Artifacts: `notes/2026-05-18-minimax-qk-helper-inplace-allreduce-negative.md`, `data/minimax-m27-qk-helper-inplace-allreduce-negative-20260518.json`, `patches/minimax-qk-helper-inplace-allreduce-negative-20260518.patch`

Recent logits-WS no-clone retie on the current no-attention-delay baseline:

- `VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1` was retested without the older final-hidden clone flag on top of the current no-attention-delay logits-WS baseline.
- It passed raw145 n64/n256 exact hashes, semantic suite, 16-repeat arithmetic, and extended sixpack.
- Result: `82.341505` output tok/s and `109.788673` total tok/s mean, essentially neutral but below the promoted `82.404268` / `109.872357` no-attention-delay baseline.
- Decision: do not promote and do not submit to LocalMaxxing. This confirms the no-clone retie is quality-safe under the current recipe, but it does not beat the promoted setting.
- Artifacts: `notes/2026-05-18-minimax-no-clone-current-baseline-neutral.md`, `data/minimax-m27-no-clone-current-baseline-neutral-20260518.json`

Recent compile allreduce custom-op screen:

- `VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1` was tested on top of the current no-attention-delay logits-WS baseline.
- It failed the first raw145 n64 exact hash check before benchmarking.
- Expected combined token hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- Candidate combined token hash: `fddec0c19f560999e0ab5c4507d694d18e71584e7d4d74342dcf24c45e567678`
- PyTorch warned that `vllm::all_reduce` output may alias an input, which is consistent with this path not being quality-safe under the current XPU graph recipe.
- Decision: do not promote and do not submit to LocalMaxxing. Keep the strict-runner env capture patch for traceability, but avoid this flag until the custom op is made alias-safe and revalidated.
- Artifacts: `notes/2026-05-18-minimax-compile-allreduce-custom-op-quality-fail.md`, `data/minimax-m27-compile-allreduce-custom-op-quality-fail-20260518.json`, `patches/minimax-strict-harness-custom-allreduce-env-capture-20260518.patch`

Recent XPU compiler-pass screens:

- `fuse_allreduce_rms` failed before raw output because the pass path asserts CUDA availability: `Torch not compiled with CUDA enabled`.
- `fuse_gemm_comms` was disabled by default for MiniMax hidden size; when forced with `sp_min_token_num=1`, it activated but failed during AOT compile with `NameError: name 'AsyncTPPass' is not defined`.
- Sequence parallelism alone activated with `sp_min_token_num=1`, but removed batch sizes `[1, 2]`, left graph capture sizes empty, and failed engine startup with `Maximum cudagraph size should be greater than or equal to 1`.
- MiniMax-specific `fuse_minimax_qk_norm` plus the XPU helper passed raw145 n64/n256 exact hashes, semantic suite, 16-repeat arithmetic, and extended sixpack.
- Result for the Q/K pass: `79.122088` output tok/s and `105.496118` total tok/s mean, below the promoted `82.404268` / `109.872357` no-attention-delay baseline.
- Decision: do not promote and do not submit to LocalMaxxing. The generic compiler/communication fusion toggles are currently CUDA-shaped or graph-shape blocked on XPU; the MiniMax Q/K pass is quality-safe but slower.
- Artifacts: `notes/2026-05-18-minimax-xpu-compiler-pass-screens.md`, `data/minimax-m27-xpu-compiler-pass-screens-20260518.json`

Recent WS top-k FP16 route-weight follow-up:

- `VLLM_XPU_MINIMAX_WS_TOPK_WEIGHT_FP16=1` kept exact MiniMax router-logits top-8 selection and score normalization, but stored the normalized route weights as FP16 before the llm-scaler INT4 work-sharing down kernel.
- It passed raw145 n64/n256 exact hashes, semantic suite, 16-repeat arithmetic, and extended sixpack with the same promoted hashes.
- Result: `81.417643` output tok/s and `108.556857` total tok/s mean, below the promoted `82.404268` / `109.872357` no-attention-delay baseline.
- Decision: do not promote and do not submit to LocalMaxxing. The route-weight dtype substitution is quality-safe but slower; keep FP32 route weights in the promoted runtime.
- Artifacts: `notes/2026-05-18-minimax-ws-topk-fp16-route-weight-negative.md`, `data/minimax-m27-ws-topk-fp16-route-weight-negative-20260518.json`, `patches/minimax-ws-topk-fp16-route-weight-negative-20260518.md`

Recent cached MoE-op lookup follow-up:

- A Python-side patch cached imported llm-scaler custom-op callables on `MoeWNA16Method` instead of dynamically importing them at the decode call boundary.
- It passed raw145 n64/n256 exact hashes, semantic suite, 16-repeat arithmetic, and extended sixpack with the same promoted hashes.
- Result: `82.006549` output tok/s and `109.342066` total tok/s mean, below the promoted `82.404268` / `109.872357` no-attention-delay baseline.
- Decision: do not promote and do not submit to LocalMaxxing. The patch was reverted from active runtime; Python import/callable lookup is not the current decode bottleneck under graph replay.
- Artifacts: `notes/2026-05-18-minimax-cached-moe-op-neutral.md`, `data/minimax-m27-cached-moe-op-neutral-20260518.json`, `patches/minimax-cached-moe-op-neutral.md`

Recent immediate MoE residual-allreduce follow-up:

- `VLLM_MINIMAX_MOE_IMMEDIATE_RESIDUAL_ALLREDUCE=1` skipped the MoE runner's standalone final allreduce and immediately folded the residual into a rank-0-residual allreduce after `block_sparse_moe`.
- It passed raw145 n64/n256 exact hashes, semantic suite, 16-repeat arithmetic, and extended sixpack with the same promoted hashes.
- Result: `82.242981` output tok/s and `109.657309` total tok/s mean, below the promoted `82.404268` / `109.872357` no-attention-delay baseline.
- Decision: do not promote and do not submit to LocalMaxxing. The runtime patch was reverted; coarse model-level residual/allreduce relocation is quality-safe but not a useful speed path.
- Artifacts: `notes/2026-05-18-minimax-moe-immediate-residual-neutral.md`, `data/minimax-m27-moe-immediate-residual-neutral-20260518.json`, `patches/minimax-moe-immediate-residual-neutral-20260518.md`

Recent CCL topology override follow-up:

- `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` was tested to bypass oneCCL fabric vertex connection checking after logs reported PCIe topology between the four B70s.
- It passed raw145 n64/n256 exact hashes, semantic suite, 16-repeat arithmetic, and extended sixpack with the same promoted hashes.
- Result: `81.736187` output tok/s and `108.981582` total tok/s mean, below the promoted `82.404268` / `109.872357` no-attention-delay baseline.
- Decision: do not promote and do not submit to LocalMaxxing. Leave this CCL env unset for the promoted four-B70 MiniMax path.
- Artifacts: `notes/2026-05-18-minimax-ccl-fabric-vertex-off-negative.md`, `data/minimax-m27-ccl-fabric-vertex-off-negative-20260518.json`

Recent logits-WS MoE-delay retest:

- `VLLM_MINIMAX_MOE_DELAY_ALLREDUCE=1` was retested on top of the current exact logits-to-work-sharing MiniMax baseline under the fair default XPU FlashAttention v2 backend.
- It passed raw145 n64/n256 exact hashes, semantic suite, 16-repeat arithmetic, and extended sixpack.
- Result: `79.019501` output tok/s and `105.359335` total tok/s mean, below the promoted `81.758267` / `109.011023` logits-WS baseline.
- Decision: do not promote and do not submit to LocalMaxxing. Coarse delayed MoE residual handling is quality-safe, but it has now lost both before and after the logits-WS promotion; the next MoE work needs kernel/epilogue or collective-shape changes, not the same allreduce relocation.
- Artifacts: `notes/2026-05-18-minimax-logits-ws-moe-delay-negative.md`, `data/minimax-m27-logits-ws-moe-delay-negative-20260518.json`

Recent logits chunked-gather rejection:

- `VLLM_XPU_LOGITS_CHUNKED_GATHER=32768` split final TP logits all-gather into smaller chunks.
- It passed raw145 n64/n256 exact hashes and the semantic suite.
- It failed the 16-repeat arithmetic gate with token-level nondeterminism: 15 repeats produced hash `def6899500b2364bc97d561fc5f9cc78aa9fbcd5a0eb032eab1f2c6735d2bbec`, while one repeat produced `9409e53d9c5444f8e179bee4951544a7b36986e5d53a0d90aca0a0479ecdecad`.
- Decision: reject without benchmarking and do not submit to LocalMaxxing.
- Artifacts: `notes/2026-05-18-minimax-logits-chunked-gather-reject.md`, `data/minimax-m27-logits-chunked-gather-reject-20260518.json`

## Qwen3.6 27B

The quality-preserving Qwen targets remain separate from MiniMax AutoRound:

- Q4_0 GGUF TP3 remains the current Qwen decode-speed focus.
- Static FP8 TP4 remains the preferred long-context Qwen layout.
- AutoRound/INT4 results should not be compared as equal-quality replacements for FP8/BF16/GGUF without separate quality validation.

## Next Optimization Targets

- Keep the MiniMax logits-to-work-sharing FlashAttention/PIECEWISE recipe as the new strict baseline.
- Target final logits/lm-head cost, hidden-state collective boundaries, MoE/projection epilogue fusion, and prefill efficiency.
- Do not promote logits/router/argmax shortcuts unless they pass the same strict quality gate.
- Avoid logits chunked-gather variants unless there is a new deterministic implementation; `VLLM_XPU_LOGITS_CHUNKED_GATHER=32768` failed repeatability.
- Keep strict runner backend defaults aligned with promoted recipes so candidate comparisons are fair.
- Keep speculative decode optional and quality-gated; no current promoted MiniMax result uses speculation.
