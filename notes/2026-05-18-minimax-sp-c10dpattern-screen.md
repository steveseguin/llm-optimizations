# MiniMax Sequence Parallel C10d Pattern Screen, 2026-05-18

## Objective

Re-test vLLM sequence parallelism on MiniMax M2.7 AutoRound INT4 TP4 after prior contaminated screens. The goal was a math-preserving communication-boundary optimization for 4x B70 decode, not a quality-changing model change.

Reference promoted strict baseline remains:

- Model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- Hardware: 4x Intel Arc Pro B70 32GB
- Context/bench shape: p512/n1536, TP4, ctx 2048, batch 1
- Output: `70.00635289969598` tok/s
- Total: `93.34180386626133` tok/s
- LocalMaxxing id: `cmpahyaas002gmn01lk0625he`
- AOT hash: `03f6a28c070656d44eab4c581bc8dc5295ed123e7c0150c7f596ea24012406b0`

## Patch

`patches/vllm-xpu-sequence-parallel-c10d-pattern-20260518.patch`

The first clean SP run failed during pattern registration because the default `tensor_model_parallel_all_reduce` pattern captured a torchbind process-group object:

```text
NotImplementedError: While executing %_torchbind_obj0 = get_attr[target=_torchbind_obj0]
```

The patch changes the SP pattern helper on XPU to express the pattern as:

```python
x = x.clone()
torch.ops._c10d_functional.all_reduce_(x, "sum", self.tp_group.unique_name)
return torch.ops._c10d_functional.wait_tensor(x)
```

That is the same functional collective shape already present in the compiled AOT graph, but keyed by group name instead of embedding the torchbind group object.

## Screens

| label | config delta | result |
| --- | --- | --- |
| `minimax-sp-clean-qualityscreen` | `pass_config.enable_sp=true`, no c10d pattern patch | startup fails with torchbind `get_attr` NotImplementedError |
| `minimax-sp-c10dpattern-qualityscreen` | c10d pattern patch, `compile_sizes=[1]`, default capture size | AOT compiles, then vLLM asserts `Maximum cudagraph size should be greater than or equal to 1` because SP removes capture size `1` under TP4 |
| `minimax-sp-c10dpattern-cg4-qualityscreen` | c10d pattern patch, `cudagraph_capture_sizes=[4]`, `compile_sizes=[1]` | AOT compiles, then vLLM rejects `compile_sizes contains 1 which would be padded to 4` |
| `minimax-sp-c10dpattern-cg4cs4-qualityscreen` | c10d pattern patch, `cudagraph_capture_sizes=[4]`, `compile_sizes=[4]` | compile reaches generated SP reduction kernels, then `ocloc`/IGC fails with `Internal Compiler Error: Floating point exception` |

The last screen is the meaningful blocker: after graph-shape fixes, current Intel compiler stack cannot compile the generated SP reduction kernel for shape `xnumel=4`, `r0_numel=1536` on BMG.

Key log lines:

```text
RuntimeError: `ocloc` failed with error code 245
IGC: Internal Compiler Error: Floating point exception
Command was: ocloc compile ... -spirv_input -device bmg
```

## Conclusion

Do not promote SP for MiniMax M2.7 AutoRound on this B70 stack. The c10d pattern patch is useful as a reproducibility artifact and gets past the first XPU-specific pattern failure, but the valid TP4 cudagraph route currently ends in an Intel compiler ICE before any quality token can be generated.

No LocalMaxxing submission was made because no run reached a quality-valid throughput benchmark.

## Follow-Up

- Keep the c10d pattern patch archived.
- Revisit SP only after an Intel compiler update or after finding an Inductor lowering/config that avoids the crashing reduction kernel.
- Prioritize a narrower XPU-specific Q/K variance or hidden-state collective fusion over full SP, since SP currently changes enough of the graph to expose compiler instability.

## Artifacts

- Clean failure summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-sp-clean-qualityscreen-strict-tp4-ctx2048-mbt512-bs256-20260518T023533Z-summary.json`
- Capture-size failure summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-sp-c10dpattern-cg4-qualityscreen-strict-tp4-ctx2048-mbt512-bs256-20260518T024611Z-summary.json`
- IGC failure summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-sp-c10dpattern-cg4cs4-qualityscreen-strict-tp4-ctx2048-mbt512-bs256-20260518T025122Z-summary.json`
