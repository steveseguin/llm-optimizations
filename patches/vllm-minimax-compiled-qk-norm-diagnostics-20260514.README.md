# vLLM MiniMax Compiled Q/K Norm Diagnostics

Patch snapshot:

`vllm-minimax-compiled-qk-norm-diagnostics-20260514.patch.gz.b64`

Decode with:

```bash
base64 -d vllm-minimax-compiled-qk-norm-diagnostics-20260514.patch.gz.b64 \
  | gzip -d > vllm-minimax-compiled-qk-norm-diagnostics-20260514.patch
```

Purpose:

- Adds finite tracing inside MiniMax attention to locate the first non-finite
  tensor under the faster TP4 `torch.compile` path.
- Adds diagnostic env guards for splitting Q/K variance allreduce and for a
  decomposed Q/K RMSNorm expression.
- Confirms that the corrupt compiled path first fails at layer 16 Q RMSNorm,
  before RoPE, attention, output projection, MoE, logits, or sampling.

This is a diagnostic patch, not a promoted speed patch. The compiled path still
fails quality gates as of this snapshot.
