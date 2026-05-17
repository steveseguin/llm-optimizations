# MiniMax No-Clone Runtime-Guard Revalidation

Date: 2026-05-17

## Result

The older `VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1` path reproduced as the new
best quality-gated 4x B70 MiniMax M2.7 AutoRound result after adding a runtime
guard for the XPU communicator patch site.

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU, TP4
- Quantization: AutoRound INT4 W4A16
- Shape: p512 / n1536 / batch 1 / context 2048
- Output throughput: `66.757143` tok/s mean
- Total throughput: `89.009524` tok/s mean
- Output repeats: `66.699729`, `66.814558` tok/s
- Total repeats: `88.932972`, `89.086077` tok/s
- Prior promoted baseline: `61.317497` output tok/s, `81.756663` total tok/s
- Uplift over prior promoted baseline: `8.87%` output tok/s
- LocalMaxxing ID: `cmp9tk7co04m3o401lhm2n9gm`

This is a valid performance result and should be shared. It is also an important
course correction: the local-argmax logits shortcut is not required for the
current best score and was slightly slower in the latest strict retest.

## Runtime Guard

The runtime guard verified that the active vLLM process imported the intended
site-packages runtime and that the no-clone marker existed in the XPU
communicator module before model load.

- `logits_processor.py` sha256:
  `c032b08de08e929fb38a54624401eeae83c618dd19edfb97e5ecdec379fbe254`
- `xpu_communicator.py` sha256:
  `9d35eb45afad1906d635f67480b15c254cbc3296ef10a8e06f977143272f43a7`
- Required-any marker:
  `VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE` found in `xpu_communicator`

The local-argmax code markers are still present in `logits_processor.py`, but
the local-argmax environment flags were intentionally unset for this run.

## Quality Gates

The result passed the strict gate before benchmarking:

- raw145 n64 exact combined token hash:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact combined token hash:
  `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: PASS prompt, arithmetic `42`, and valid `add_one` function
- arithmetic repeat suite: 8 greedy repeats, deterministic, all matched `42`

## Interpretation

The previous local-argmax focused result was correct, but the faster path is to
leave logits behavior closer to the stock vLLM path and keep the no-clone XPU
communicator optimization active. This suggests the next work should focus on
TP communication scheduling and communicator overhead, not another logits-stage
gather wrapper.

The `minimaxlogits + local_argmax + no_clone` retest also passed quality, but
only reached `61.217915` output tok/s and `81.623886` total tok/s. That makes it
a valid no-improvement candidate rather than a shareable improvement.

## Artifacts

- Summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-no-clone-oldenv-runtimeguard-repro-strict-tp4-ctx2048-mbt512-bs256-20260517T131435Z-summary.json`
- Benchmark JSONs:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T132717Z.json`
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T133008Z.json`
- Data summary:
  `data/minimax-m27-no-clone-oldenv-runtimeguard-repro-20260517.json`
- LocalMaxxing payload:
  `data/localmaxxing-minimax-m27-autoround-no-clone-oldenv-runtimeguard-repro-p512n1536-20260517.payload.json`
- LocalMaxxing response:
  `data/localmaxxing-responses/minimax-m27-autoround-no-clone-oldenv-runtimeguard-repro-p512n1536-20260517.response.json`
