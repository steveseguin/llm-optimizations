# 2026-05-08 MiniMax K/Q/V Offload Layer Improvement

## Result

After the RPC device-map fix, retesting K/Q/V offload produced a clean quality-preserving MiniMax improvement:

- Model: `MiniMaxAI/MiniMax-M2.7`, local Unsloth GGUF `UD-IQ4_XS`
- Hardware: `4x Intel Arc Pro B70 32GB`
- Engine: `ik_llama.cpp` RPC client plus four SYCL Level Zero workers
- Baseline: `14.292387 tok/s`, `-nkvo 1`
- Improved: `16.383602 tok/s`, `-nkvo 0`
- Improvement: about `14.6%`
- LocalMaxxing: `cmowft2hr000oo3019is4snoq`, `APPROVED`

This does not change target model quality. KV cache remains `f16`; the change enables K/Q/V offload instead of forcing `offload_kqv=false`. No speculative decoding, no expert dropping, no power-limit changes.

## Repro

```bash
GGML_DISABLE_FUSED_RMS_NORM=1 \
GGML_DISABLE_FUSED_MUL_UNARY=1 \
/home/steve/src/ik_llama.cpp/build-rpc-client-cpu/bin/llama-bench \
  -m /home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -rpc '127.0.0.1:50100|0,127.0.0.1:50101|0,127.0.0.1:50102|0,127.0.0.1:50103|0' \
  -p 0 -n 64 -r 3 -ngl 99 -sm layer -ts 1/1/1/1 \
  -fa 0 -nkvo 0 -ub 32 -ctk f16 -ctv f16 -t 4 -w 0 -mmp 1 \
  -rtr 1 -fmoe 1 -no-mmad 0 -muge 1 -sas 0 -o json
```

## Runs

| Config | tok/s | Samples | Notes |
| --- | ---: | --- | --- |
| `-nkvo 1`, r3 | `14.292387` | `13.8745`, `14.4737`, `14.529` | Previous corrected baseline |
| `-nkvo 0`, r1 | `15.642624` | n/a | Clean smoke after device-map fix |
| `-nkvo 0`, r3 | `16.383602` | `16.2585`, `16.4391`, `16.4532` | New best layer baseline |

Raw data:

```text
data/minimax-m27-kqv-offload-layer-improvement-20260508.json
data/localmaxxing-submission-minimax-m27-kqv-offload-20260508.json
```

## Interpretation

The earlier `-nkvo 0` attempts were not meaningful because they were made around the broken RPC device mapping and old split string. With the corrected `-ts 1/1/1/1` mapping, K/Q/V offload is stable and faster.

The timing probe from the previous note showed attention decode and KV movement as major layer-mode costs. This result confirms that the K/Q/V placement flag was leaving performance on the table.

## Next

1. Treat `16.383602 tok/s` as the current MiniMax UD-IQ4_XS 4x-B70 baseline.
2. Sweep `-nkvo 0` with a small set of otherwise low-risk flags: `-sas 1`, `-ub 16/64`, and possibly `-ctk q8_0 -ctv q8_0` as a quality-tradeoff experiment only.
3. Keep graph branch-fusion as diagnostic; it remains much slower than this layer path.
