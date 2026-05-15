# LocalMaxxing Submissions

Date: 2026-05-15

Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`, AutoRound W4A16 safetensors,
vLLM/XPU TP4.

| Label | LocalMaxxing ID | GPUs | Input | Output | tok/s out | tok/s total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `vllm-minimax-m27-clean-weight-piecewise-aot-p512-n1536` | `cmp6a5c1o00mpo3011hg8ncyp` | 4 | 512 | 1536 | 65.752 | 87.670 |

Note: repaired piecewise/AOT compiled path with the default-off MiniMax Q/K
RMSNorm clean-weight guard enabled. Three p512/n1536 repeats were `64.622`,
`66.659`, and `65.976` output tok/s. Raw-prompt quality canaries at 64 and
256 generated tokens both passed with `0` NUL tokens, `0` non-space control
chars, and nontrivial token diversity. This supersedes the earlier quality-
corrected `~61` tok/s TP4 baseline, but the older `~73` tok/s AOT diagnostic
remains invalid because it failed the raw corruption gate.

Date: 2026-05-09

Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`, AutoRound W4A16 safetensors, vLLM/XPU TP4.

| Label | LocalMaxxing ID | GPUs | Input | Output | tok/s out | tok/s total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `vllm-minimax-m27-autoround-u4-decode-p512-n128` | `cmoxptkfd00hsml01hf2ajhhp` | 4 | 512 | 128 | 29.748 | 148.742 |
| `vllm-minimax-m27-autoround-u4-decode-p512-n256` | `cmoxq7cww00i8ml019ihbeqc9` | 4 | 512 | 256 | 33.034 | 99.101 |
| `vllm-minimax-m27-autoround-u4-fp32-route-p512-n256` | `cmoy8hs3n002smk01ksgcpavr` | 4 | 512 | 256 | 34.158 | 102.474 |
| `vllm-minimax-m27-autoround-u4-pp2tp2-negative-p512-n256` | `cmoy9exmf003lmk01d3it9cz2` | 4 | 512 | 256 | 17.550 | 52.651 |
| `vllm-minimax-m27-autoround-u4-default-ipc-p512-n256` | `cmoy9qat60040mk01l5y8n3al` | 4 | 512 | 256 | 34.578 | 103.734 |
| `vllm-minimax-m27-autoround-u4-default-ipc-p512-n512` | `cmoyagit0004dmk014gk25e2k` | 4 | 512 | 512 | 37.136 | 74.272 |
| `vllm-minimax-m27-autoround-xpu-graph-fixedkv-p512-n256` | `cmoyfl7cm0057mk01suxo0glp` | 4 | 512 | 256 | 32.723 | 98.169 |

Note: unsigned llm-scaler u4 decode-only MoE path, no speculative decode, no expert dropping, no sampling changes, and no power-limit changes. The XPU graph fixed-KV result is a negative/diagnostic run: PIECEWISE graph capture succeeded with local vLLM patches, but it was slower than the non-graph default-IPC path.

Date: 2026-05-03

Model: `Lorbus/Qwen3.6-27B-int4-AutoRound`

All submitted results returned `APPROVED`.

| Label | LocalMaxxing ID | GPUs | Input | Output | tok/s out | tok/s total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `vllm-int4-single-b70-mtp-500-256` | `cmoq41b9d001alg043wsnthz2` | 1 | 500 | 256 | 45.2 | 133.44 |
| `vllm-int4-single-b70-mtp-500-512` | `cmoq47sll0005l104v3i0f9l3` | 1 | 500 | 512 | 41.3 | 81.60 |
| `vllm-int4-tp2-b70-nonmtp-500-256` | `cmoq4e9dw0002js04ledqyycn` | 2 | 500 | 256 | 49.1 | 144.88 |
| `vllm-int4-tp2-b70-nonmtp-500-512` | `cmoq4krfb000cl40456wobg7e` | 2 | 500 | 512 | 48.3 | 95.56 |
| `vllm-int4-single-b70-nonmtp-500-256` | `cmoq4r8rc0001l804tocgibus` | 1 | 500 | 256 | 31.8 | 93.80 |
| `vllm-int4-tp2-b70-mtp-500-256` | `cmoq4xppt0003ky04xidngli9` | 2 | 500 | 256 | 35.6 | 105.03 |
