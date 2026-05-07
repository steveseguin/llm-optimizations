# llama.cpp Qwen35 Fused Beta/Alpha Experimental Patch

Patch artifact:

- `llama-cpp-qwen35-fused-beta-alpha-experimental-20260507.patch.gz.b64`
- decoded patch sha256: `1aea32ffd0318fdaf8eef98bc914f433226cf6d852cac521fe85e9c2a1203409`
- encoded artifact sha256: `38198d1bf72b0bbb3598c0ab0933738e0dd365bed4a62f0e8dcb2628bce2f8c5`

Decode:

```bash
base64 -d llama-cpp-qwen35-fused-beta-alpha-experimental-20260507.patch.gz.b64 | gunzip > llama-cpp-qwen35-fused-beta-alpha-experimental-20260507.patch
```

Scope:

- optional Qwen35/Qwen35MoE `blk.N.ssm_ba.weight` loader path;
- Qwen35 graph path for fused beta/alpha projection;
- Qwen35 fused tensor split-granularity fix;
- Meta split-state diagnostics needed to debug the first failed fused load.

Status:

- speed-positive on TP3 Q4_0 GGUF;
- not quality-cleared because full logits differ from the original model;
- do not treat as a production patch or LocalMaxxing result yet.
