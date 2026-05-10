# MiniMax Inline Linear Attention Check, 2026-05-10

Hypothesis: vLLM wraps `MiniMaxText01LinearAttention.forward()` behind the
opaque `torch.ops.vllm.linear_attention` custom op. If the active MiniMax
checkpoint used that path, inlining the same math behind a default-off env gate
could potentially remove a per-layer no-compile boundary.

Result: not applicable to `Lasimeri/MiniMax-M2.7-int4-AutoRound`. The active
model uses `vllm/model_executor/models/minimax_m2.py` normal attention, not
`MiniMaxText01LinearAttention`.

Smoke details:

- Env gate tested: `VLLM_MINIMAX_INLINE_LINEAR_ATTENTION=1`.
- Tiny p1/n8 smoke compiled and generated.
- Full-shape p512/n512 cold isolated-cache run produced the same AOT hash as
  the normal path: `4799a3c8468de261861723fba07480ef61e010f504245a62e5e93f4e9aef8e22`.
- The p512/n512 run showed the usual cold-AOT artifact: `9,408` KV tokens and
  `27.814660` output tok/s.
- Logs:
  - `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-inline-linear-attn/vllm-minimax-m27-autoround-tp4-p1n8-20260510T115940Z.log`
  - `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-inline-linear-attn/vllm-minimax-m27-autoround-tp4-p512n512-20260510T120244Z.log`

Conclusion: do not spend more time on `MiniMaxText01LinearAttention` for this
model. The default-off gate was removed from both the source tree and installed
vLLM package after the check.
