# MiniMax DFlash Fast-NVMe Retest, 2026-05-10

## Context

After moving the MiniMax AutoRound path to `/mnt/fast-ai`, I retested the
`MirecX/MiniMax-M2.7-L3H5-DFlash` drafter from fast NVMe rather than the prior
external-drive location. The purpose was to check whether the previous DFlash
stall was storage/setup related before deprioritizing speculative decoding in
favor of target-model fusion.

## Command Shape

```bash
MODEL=/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround
TP=4
MAX_MODEL_LEN=512
MAX_BATCHED_TOKENS=128
MAX_NUM_SEQS=1
INPUT_LEN=64
OUTPUT_LEN=32
NUM_PROMPTS=1
DTYPE=float16
USE_LLM_SCALER_MOE=1
CCL_P2P=1
XPU_GRAPH=0
EXTRA_ARGS='--speculative-config {"method":"dflash","model":"/mnt/fast-ai/llm-models/minimax-m2.7-l3h5-dflash","num_speculative_tokens":4,"draft_tensor_parallel_size":1}'
```

Log:

```text
/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-dflash-smoke/vllm-minimax-m27-autoround-tp4-p64n32-20260510T155100Z.log
```

## Result

Negative / no LocalMaxxing submission.

The storage move did not fix the speculative path. The target model and DFlash
drafter loaded and compiled successfully:

- target model load: `28.4 GiB`, `71.897575 s`
- drafter checkpoint size: `0.70 GiB`
- selected auxiliary layers: `(2, 16, 30, 43, 57)`
- target AOT key:
  `ccf3b6d8830c47d11a03c6d3fdbb89d41282db1e3fad08340b43fa8f6c08c8c9`
- DFlash head AOT key:
  `bec81810493fe306045a8a1a9f7df60c42624e6750d736390d09857e06a4c4a6`
- reported GPU KV cache: `4,160` tokens at `max_model_len=512`

Generation then stalled at:

```text
Processed prompts: 0/1, est. speed input: 0.00 toks/s, output: 0.00 toks/s
```

vLLM emitted repeated shared-memory broadcast warnings after 60-second waits.
The run was terminated and no benchmark JSON was produced.

## Decision

Do not spend more long benchmark windows on MiniMax DFlash on this runtime. It
is target-verified if it works, so quality would be preserved, but the current
XPU TP4 harness still cannot produce a tiny p64/n32 result. Keep the drafter
downloaded for capped future smoke tests after a vLLM/runtime change, and
prioritize quality-preserving target-model fusion:

- Q/K RMS variance plus TP collective boundaries;
- output projection / MoE allreduce epilogues;
- attention/KV scheduling and graph-cache behavior.
