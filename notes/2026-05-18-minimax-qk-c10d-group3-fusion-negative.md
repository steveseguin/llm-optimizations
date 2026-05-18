# MiniMax Q/K RMS C10D Group-3 Fusion Negative

Date: 2026-05-18

## Summary

The MiniMax Q/K RMS fusion pass was updated to match the actual compiled XPU
graph shape: non-mutating `_c10d_functional.all_reduce(..., "sum", "3")`
followed by `_c10d_functional.wait_tensor`. With
`VLLM_MINIMAX_QK_NORM_XPU_HELPER_FUSION=1` and
`VLLM_MINIMAX_QK_NORM_C10D_GROUP_NAME=3`, the pass replaced 62 patterns and
removed the Q/K RMS variance collectives from the AOT decode graph.

This is a real graph simplification, but it is not a performance win yet.
The strict-quality known-good AOT cache passed every gate, then two valid
p512/n1536 benchmark repeats averaged `68.214061` output tok/s and
`90.952082` total tok/s. That is `-2.560184%` versus the current promoted
FlashAttention/PIECEWISE strict baseline (`70.006353` output tok/s). A third
repeat loaded the same AOT hash but hit the shared-memory broadcast stall guard.

Do not submit this as a LocalMaxxing achievement. Keep it as a documented
negative result and as a useful graph-pattern patch for future fusion work.

## Recipe

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Engine: vLLM `0.20.1-local`, XPU TP4
- Hardware: 4x Intel Arc Pro B70 32GB
- Quantization: AutoRound INT4 W4A16 / INC
- Attention backend: default XPU FlashAttention v2
- Shape: p512, n1536, ctx2048, batch 1
- Dtype: `float16`
- Block size: 256
- Max batched tokens: 512
- Prefix cache: disabled
- Temperature: greedy / 0
- AOT hash:
  `7f1422b4de9682e60e5291b8434407b8f49e38907b85537ba584087183bfb1bf`
- Cache root:
  `/mnt/fast-ai/vllm-cache-exp/minimax-qk-c10d-group3-n256-freshonly-20260518T042509Z`

Important env:

- `VLLM_XPU_ENABLE_XPU_GRAPH=1`
- `VLLM_XPU_FORCE_GRAPH_WITH_COMM=1`
- `VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE=1`
- `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1`
- `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1`
- `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2`
- `VLLM_MINIMAX_QK_NORM_XPU_HELPER_FUSION=1`
- `VLLM_MINIMAX_QK_NORM_C10D_GROUP_NAME=3`
- `VLLM_MINIMAX_QK_NORM_PASS_LOG=1`
- `CCL_TOPO_P2P_ACCESS=1`
- `ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3`
- `ZE_AFFINITY_MASK=0,1,2,3`

Benchmark extra args:

```bash
--async-engine \
--block-size 256 \
--no-enable-prefix-caching \
--compilation-config '{"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE","pass_config":{"fuse_minimax_qk_norm":true}}'
```

## Quality

Full strict quality passed when reusing the known-good fused AOT cache:

- raw145 n64 exact:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact:
  `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: PASS
- arithmetic repeat: exact `42`, 16 greedy repeats, deterministic
- extended sixpack: PASS

This means the fused AOT artifact can be quality-correct. It does not mean the
fresh AOT compile path is fully reliable.

## Reliability Finding

A first full strict run with a fresh cache failed the raw145 n256 exact gate
after passing raw145 n64. The failed output began by copying prompt text:

```text
Answer with four concise bullets about PCIe multi GPU inference. alpha beta gamma...
```

The failed cache and the known-good cache used the same AOT hash
`7f1422b4de9682e60e5291b8434407b8f49e38907b85537ba584087183bfb1bf`, but the
rank model file hashes differed.

Known-good rank model SHA256 values:

- rank 0: `4daacd2db21b48cdaa55278ad5cacbc1c7413cf1db1bd9cb2c3dcb74270b9d3c`
- rank 1: `5e7fac18193ee4882e01ecfb689839975c847310e65525aeec8c0cff11dc512b`
- rank 2: `fe7a255bc0abe68bfbaf49951e9e6971ab3efeb749d0350b5b0cb3f93d36643b`
- rank 3: `ecd1156e3ac8379e19decac05424dbbb7884cdc8fde371c13cde959ff517f07e`

Failed fresh-cache rank model SHA256 values:

- rank 0: `b1121fd8be7fef67b730b797db62fa7e4e5ebcbeeaf8776f29c8bf211fce38f8`
- rank 1: `15d8e07369e76a6c097e6a420c655547c745a32d68258df559ec44bc23cf6edf`
- rank 2: `cb2b40e3a227ad4f76365bb4eea37be64b0c545c2631be9889326939de814b7c`
- rank 3: `ae868c91cbb75f2585598648de19c9d055ff6b22554d7b2f19cafa2e6b09b395`

This suggests AOT artifact generation can differ under the same hash. Treat
fresh-cache Q/K fusion results with suspicion until this is understood.

## Results

Valid benchmark repeats:

| repeat | elapsed s | output tok/s | total tok/s |
| --- | ---: | ---: | ---: |
| 1 | 22.530215 | 68.175114 | 90.900152 |
| 2 | 22.504502 | 68.253009 | 91.004011 |
| mean | - | 68.214061 | 90.952082 |

The third repeat loaded the fused AOT hash but stalled after `torch.compile`
load, then the shared-memory stall guard terminated it after three warnings.

Baseline comparison:

- promoted strict baseline mean output tok/s: `70.006353`
- fused Q/K C10D group-3 mean output tok/s: `68.214061`
- delta: `-2.560184%`

## AOT And Collectives

Baseline FlashAttention/PIECEWISE AOT hash
`03f6a28c070656d44eab4c581bc8dc5295ed123e7c0150c7f596ea24012406b0` had:

- actual allreduce call lines: `1496`
- actual wait tensor call lines: `1496`
- embedding hidden: `8`
- Q/K RMS variance: `496`
- attention output projection hidden: `496`
- MoE hidden: `496`

Fused C10D group-3 AOT hash
`7f1422b4de9682e60e5291b8434407b8f49e38907b85537ba584087183bfb1bf` had:

- actual allreduce call lines: `1000`
- actual wait tensor call lines: `1000`
- embedding hidden: `8`
- Q/K RMS variance: `0`
- attention output projection hidden: `496`
- MoE hidden: `496`

This confirms the pass removed the target collectives, but those 496 small
variance allreduces were not the dominant wall-time bottleneck in this recipe.

## Decision

Reject this as a promoted speed path. Keep the patch and artifacts because they
prove the actual XPU C10D pattern shape and give us a reliable way to remove
Q/K variance collectives for later fusion experiments.

The next higher-value work is still the remaining 1000 hidden-state collectives:
attention output projection and MoE hidden allreduce boundaries. Fusion around
those boundaries is more likely to affect decode wall time than removing the
tiny Q/K variance collectives alone.

## Artifacts

- Strict quality pass summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-qk-c10d-group3-aot-goodcache-strict-quality-20260518T045103Z-strict-tp4-ctx2048-mbt512-bs256-20260518T045103Z-summary.json`
- Strict quality pass directory:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-qk-c10d-group3-aot-goodcache-strict-quality-20260518T045103Z-strict-tp4-ctx2048-mbt512-bs256-20260518T045103Z-quality`
- Initial failed fresh-cache summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-qk-c10d-group3-strict-quality-20260518T041513Z-strict-tp4-ctx2048-mbt512-bs256-20260518T041513Z-summary.json`
- Benchmark summary:
  `/home/steve/bench-results/minimax-m2.7-autoround-vllm/qk-c10d-group3-aot-goodcache-bench-20260518T050352Z.summary.log`
- Benchmark JSON 1:
  `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T050352Z.json`
- Benchmark JSON 2:
  `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T050634Z.json`
- Stall log:
  `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T050921Z.log`
- Patch:
  `patches/vllm-minimax-qk-c10d-group3-pattern-negative-20260518.patch`
