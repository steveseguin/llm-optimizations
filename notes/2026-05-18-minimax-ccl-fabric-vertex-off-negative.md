# MiniMax M2.7 CCL Fabric Vertex Override Retest

Date: 2026-05-18

## Summary

`CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` was tested because oneCCL logs report PCIe topology between the four B70s and mention this switch as a way to bypass fabric vertex connection checking. This is a software-only communication-stack candidate; it does not change model math.

Candidate recipe:

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Baseline recipe: FP16 activations, AutoRound INT4 W4A16, default XPU FlashAttention v2, PIECEWISE XPU graph, exact MiniMax router logits feeding llm-scaler INT4 MoE work-sharing, `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0`
- Added env: `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`
- Shape: p512/n1536, ctx2048, batch 1, MBT512, block256

Result:

- Candidate: `81.736187` output tok/s, `108.981582` total tok/s, mean of two gated benchmark repeats
- Promoted baseline: `82.404268` output tok/s, `109.872357` total tok/s, mean of four clean long repeats
- Delta: `-0.81%` output tok/s, `-0.81%` total tok/s

Decision: do not promote and do not submit to LocalMaxxing.

## Quality Gate

The candidate passed:

- raw145 n64 exact: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: `adacbf144264486ea7d378ebb6a4c0ba23951b72c4cf86251a762b07ebef5805`
- 16-repeat arithmetic gate: `578ec378bf31cb16fb49ac5c0043270fd00a0f7898e18ac498a41ffe775d7994`
- extended sixpack: `1e3560554f57b2b56cec8f49f28bc8ba12e9e0ced26bdc99a976f1433c99caa7`

## Artifacts

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-ccl-fabric-vertex-off-ar-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T211409Z-summary.json`
- Bench JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T212956Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T213251Z.json`

## Learning

For this four-B70 layout, disabling CCL fabric vertex connection checking is quality-safe but slower. Keep the promoted default for this variable unset. Future communication work should focus on reducing collective boundaries or fusing useful work around them, not this topology override.
