# MiniMax M2.7 DFlash Speculative Decode Blocker

Date: 2026-05-09

Target model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`

Drafter: `MirecX/MiniMax-M2.7-L3H5-DFlash`, downloaded to `/mnt/corsair-external/llm-models/minimax-m2.7-l3h5-dflash`

Runtime: vLLM/XPU TP4 on 4x Arc Pro B70, `USE_LLM_SCALER_MOE=1`, `DTYPE=float16`, `XPU_GRAPH=0`, default scalar llm-scaler u4 MiniMax decode path.

## Why This Was Tested

The current best MiniMax AutoRound path is already over 30 tok/s with no speculative decoding:

- p512/n256: `34.578045` output tok/s
- p512/n512: `37.136187` output tok/s

DFlash looked worth a smoke because it is a native vLLM speculative path for MiniMax-style drafters and the local model card provides the expected target taps. It was not expected to be a guaranteed win: the card reports `m_accept ~= 1.38`, explicitly below the approximate break-even point for Strix Halo TP4, and our target is the AutoRound INT4 checkpoint rather than the AWQ-4bit target named on the drafter card.

## Command

```bash
USE_LLM_SCALER_MOE=1 \
CCL_IPC=default \
XPU_GRAPH=0 \
DTYPE=float16 \
INPUT_LEN=64 \
OUTPUT_LEN=16 \
MAX_MODEL_LEN=512 \
MAX_BATCHED_TOKENS=256 \
MAX_NUM_SEQS=1 \
NUM_PROMPTS=1 \
TP=4 \
EXTRA_ARGS='--speculative-config {"method":"dflash","model":"/mnt/corsair-external/llm-models/minimax-m2.7-l3h5-dflash","num_speculative_tokens":4}' \
/home/steve/llm-optimizations-publish/scripts/bench-vllm-minimax-autoround-xpu.sh
```

## Observed Behavior

The DFlash path loaded and initialized far enough to prove vLLM compatibility at model-registration level:

- vLLM resolved the target architecture as `MiniMaxM2ForCausalLM`.
- vLLM resolved the drafter architecture as `DFlashDraftModel`.
- The target model loaded across TP4 from the external NTFS drive.
- The drafter loaded successfully.
- vLLM shared target `embed_tokens` and `lm_head` with the drafter.
- vLLM selected auxiliary target layers `(2, 16, 30, 43, 57)`.
- Backbone and drafter/eagle-head torch compile completed.
- Reported model memory was `28.41 GiB` and available KV cache was `0.26 GiB` for the deliberately tiny `max_model_len=512` smoke.

The first request did not complete:

- prompt rendering completed;
- generation stayed at `Processed prompts: 0/1`;
- no JSON throughput file was produced;
- after about 60 seconds inside generation, vLLM logged:
  `No available shared memory broadcast block found in 60 seconds`;
- all four worker processes remained busy;
- the run was stopped and the orphaned workers were killed.

Log:

- `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260509T185848Z.log`

## Conclusion

Do not treat this DFlash drafter as a near-term MiniMax speed path on the current B70 XPU stack. It is useful as a compatibility lead, but the first smoke stalls before producing 16 tokens, and the drafter's own reported acceptance is too low to expect a speedup even after the stall is fixed.

This was not submitted to LocalMaxxing because it produced no valid throughput metric.

## Follow-up Ideas

- Only revisit DFlash after isolating the XPU speculative loop stall, preferably with a much smaller target first.
- If DFlash is retested on MiniMax, try `num_speculative_tokens=1` or `2` as a bug-isolation setting rather than a speed setting.
- A useful speed-oriented speculative path likely needs a drafter trained/calibrated against the exact AutoRound or active target checkpoint, with materially higher acceptance than `1.38`.
- Keep prioritizing quality-preserving non-speculative work: XPU equivalent of CUDA MiniMax Q/K allreduce+RMS fusion, a more monolithic routed-expert decode op, and a BF16-capable tiny llm-scaler kernel.
