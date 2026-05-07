# llama.cpp Qwen35 Fused Beta/Alpha Experimental Patch

Patch artifact:

- `llama-cpp-qwen35-fused-beta-alpha-experimental-20260507.patch.gz.b64`
- decoded patch sha256: `df786dd95dee04f072029ddcc1f054c2816b339d0d76c433fa86eda49a5054aa`
- encoded artifact sha256: `7a19e61fa0bc9923e1b0bb6f4e0ea4404fe402f54d9411a6cbb5fb2118226ac3`
- decoded patch line count: `1343`

Decode:

```bash
base64 -d llama-cpp-qwen35-fused-beta-alpha-experimental-20260507.patch.gz.b64 | gunzip > llama-cpp-qwen35-fused-beta-alpha-experimental-20260507.patch
```

Scope:

- optional Qwen35/Qwen35MoE `blk.N.ssm_ba.weight` loader path;
- Qwen35 graph path for fused beta/alpha projection;
- Qwen35 fused tensor split-granularity fix;
- Meta split-state propagation needed to preserve fused `ssm_ba` row ownership through `MUL_MAT` and exact beta/alpha `VIEW` subsets;
- current in-file dependencies from the active local Q4_0 SYCL patch stack where they share the same touched files.

Status:

- quality-cleared only with `GGML_SYCL_COMM_FUSEADD_ROOT_RESIDUAL=0`;
- final no-root TP3 result: `50.129900 tok/s` decode at 512 prompt / 512 output;
- root-residual plus meta allreduce-add is not quality-cleared and remains the next bug to fix before this can become a production patch.
