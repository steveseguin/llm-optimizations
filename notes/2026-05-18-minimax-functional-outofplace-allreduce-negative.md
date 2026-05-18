# MiniMax Functional Out-Of-Place Allreduce Negative

Date: 2026-05-18

## Summary

Tested the XPU communicator functional collective path with:

- `VLLM_XPU_COMPILE_OUT_OF_PLACE_ALLREDUCE=1`
- `VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=0`
- `VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=0`
- `VLLM_XPU_USE_LLM_SCALER_MOE=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`
- `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0`
- `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1`
- `VLLM_XPU_ENABLE_XPU_GRAPH=1`
- `VLLM_XPU_FORCE_GRAPH_WITH_COMM=1`
- `VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1`
- `CCL_TOPO_P2P_ACCESS=1`

This uses `torch.distributed._functional_collectives.all_reduce` inside `XpuCommunicator.all_reduce()` during compile, making the allreduce out-of-place rather than using the custom alias-sensitive op.

## Quality

The candidate passed the full strict quality gate:

- raw145 n64 exact hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact hash: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite hash: `adacbf144264486ea7d378ebb6a4c0ba23951b72c4cf86251a762b07ebef5805`
- 16-repeat arithmetic hash: `578ec378bf31cb16fb49ac5c0043270fd00a0f7898e18ac498a41ffe775d7994`
- extended sixpack hash: `1e3560554f57b2b56cec8f49f28bc8ba12e9e0ced26bdc99a976f1433c99caa7`

This is important because the earlier `MAX_BATCHED_TOKENS=768` retie failed the extended sixpack on the sort/list prompt. The functional allreduce candidate did not show that nondeterminism.

## Performance

Shape: p512/n1536, ctx2048, batch 1, TP4, MBT512, block256.

Two repeats:

- `82.461324` output tok/s, `109.948432` total tok/s
- `82.114830` output tok/s, `109.486440` total tok/s

Mean:

- `82.288077` output tok/s
- `109.717436` total tok/s

Comparison:

- Current promoted clone-safe custom allreduce: `87.279129` output tok/s, `116.372172` total tok/s
- Functional out-of-place allreduce delta: `-5.72%` output tok/s versus promoted

## Decision

Do not promote and do not submit to LocalMaxxing. The path is quality-safe and reproducible enough to keep as an alternative debugging baseline, but it is slower than the clone-safe compiled custom allreduce path.

The benchmark logs still show the known post-shutdown `Bad address (src/pipe.cpp:367)` line from the multiprocessing/ZeroMQ teardown path. Both benchmark repeats wrote JSON, reported `Exit status: 0`, and were included in the summary, so this is recorded as a teardown warning rather than a failed run.

## Artifacts

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-functional-outofplace-allreduce-ar-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T233441Z-summary.json`
- Quality dir: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-functional-outofplace-allreduce-ar-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T233441Z-quality`
- Benchmark JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T235023Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T235315Z.json`
