# MiniMax Q/K Helper Current-Recipe Retest, 2026-05-13

## Purpose

Retest the default-off MiniMax Q/K helper fusion inside the current best recipe,
because the earlier rejection happened before the later XPU graph, block-size,
max-batched-token, and prefix-cache-off wins.

Quality guardrail: this did not change the model, quantization, dtype, KV dtype,
sampler, tensor-parallel degree, expert routing, or allreduce semantics.

## Setup

Baseline recipe:

- TP4 FP16 on 4x Intel Arc Pro B70
- llm-scaler INT4 MoE
- XPU graph with graph partition and `compile_sizes=[1]`
- MiniMax attention delayed allreduce
- `--block-size 256`
- `MAX_BATCHED_TOKENS=512`
- `--no-enable-prefix-caching`

Retest additions:

```bash
VLLM_MINIMAX_QK_NORM_XPU_HELPER_FUSION=1
--compilation-config '{"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE","pass_config":{"fuse_minimax_qk_norm":true}}'
```

Cache root:

```text
/mnt/fast-ai/vllm-cache-exp/minimax-xpugraph-attndelay-qkhelper-block256-mbt512-noprefix-20260513T194500Z
```

AOT hash:

```text
7da4e01598f805f5c4a9841c387d77f9956b1771eb9e78838a768d330a4c19dc
```

## Results

| Run | Output tok/s | Total tok/s | Decision |
| --- | ---: | ---: | --- |
| current best p512/n1536 | `73.306312` | `97.741749` | promoted baseline |
| Q/K helper warmup p512/n128 | `16.546109` | `82.730546` | slow warmup |
| Q/K helper p512/n1536 | `73.126139` | `97.501519` | close but negative |
| Q/K helper p512/n1536 repeat | `72.282518` | `96.376691` | negative |

The first sustained run was only `0.180173` output tok/s below the current
best, so I repeated it against the warmed AOT cache. The repeat widened the
gap to `1.023794` output tok/s below current best.

## AOT Collective Check

The Q/K helper AOT graph still shows the same TP communication structure:

- actual allreduce lines: `1496`
- waits: `1496`
- wait gap: all waits were exactly 2 generated-code lines after allreduce
- categories:
  - embedding hidden allreduce: `8`
  - Q/K RMS variance allreduce: `496`
  - attention output projection hidden allreduce: `496`
  - MoE hidden allreduce: `496`

That explains the result. The helper changes local Q/K variance/apply plumbing,
but it does not remove or fuse the real collective boundary.

## Decision

Keep the Q/K helper fusion disabled for the production recipe. It remains a
useful correctness scaffold for future fused XPU collective work, but the
current implementation is not the path to a higher sustained decode ceiling.

No LocalMaxxing submission: this was valid and quality-preserving, but it did
not beat the public/current best.

## Artifacts

- Data summary:
  `data/minimax-m27-qk-helper-current-recipe-retest-20260513.json`
- First p512/n1536 log:
  `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T194704Z.log`
- Repeat p512/n1536 log:
  `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T195543Z.log`
