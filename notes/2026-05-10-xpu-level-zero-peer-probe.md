# XPU Level Zero Peer Probe for B70 Q/K Fusion

## Summary

The four B70s report Level Zero peer access across every source/destination
pair. A small standalone probe also verified both same-process P2P remote writes
and forked-process IPC remote writes for all 16 pairs.

This is important for MiniMax M2.7 because it means a persistent XPU
peer-visible workspace is viable for a future equivalent of vLLM's CUDA
`minimax_allreduce_rms_qk` Lamport path.

## Findings

- Level Zero sees one driver and four Intel GPUs: `Intel(R) Graphics [0xe223]`.
- Each device reports `maxAllocMiB=31023`, consistent with 32 GB class VRAM and
  the current driver allocation limit.
- External memory import/export support is `DMA_BUF` on all four devices.
- `zeDeviceCanAccessPeer` succeeds for every pair.
- `zeDeviceGetP2PProperties` reports:
  - self-pairs: `ACCESS|ATOMICS`;
  - cross-card pairs: `ACCESS`;
  - cross-card remote atomics are not advertised.
- `--p2p-fill-test` passed for all 16 source/destination pairs.
- `--ipc-fork-test` passed for all 16 source/destination pairs: parent exported
  a Level Zero IPC handle, a clean child process opened it on the source GPU,
  filled through that pointer, and the owner GPU verified the pattern.

## Reproduction

```bash
cd /home/steve/llm-optimizations-publish/experiments/xpu_level_zero_peer_probe
g++ -std=c++17 -O2 peer_probe.cpp -lze_loader -o peer_probe
./peer_probe --ipc-fork-test
```

## Impact

This removes a major feasibility concern for the XPU MiniMax Q/K fusion. The
next prototype should allocate a per-rank Level Zero/SYCL device workspace,
export/open peer handles across TP worker processes, and pass a device-side
pointer table into a SYCL kernel. Because cross-card atomics are unavailable,
the first design should mirror the CUDA Lamport idea with per-rank slots and
sequence/sentinel values rather than relying on remote atomic add/CAS.

The first target remains Q/K variance exchange plus RMS apply for MiniMax decode
tokens. RoPE/KV fusion should wait until the peer workspace protocol is correct
and faster than the existing oneCCL path.

