# XPU Level Zero Peer Probe

Small standalone probe for the MiniMax XPU Q/K allreduce+RMS fusion work. It
queries Level Zero device peer access, remote atomics, and external memory
import/export support for the visible Intel GPUs.

Build and run:

```bash
g++ -std=c++17 -O2 peer_probe.cpp -lze_loader -o peer_probe
./peer_probe
./peer_probe --p2p-fill-test
./peer_probe --ipc-fork-test
```

The intended use is to decide whether a persistent peer-visible workspace is
viable before prototyping an XPU equivalent of vLLM's CUDA
`minimax_allreduce_rms_qk` Lamport path.

`--p2p-fill-test` allocates a small buffer on each peer device and asks every
source device to fill that peer allocation through a Level Zero command list.
It then copies the buffer back from the owner device and verifies the pattern.
This is a cheap functional test for remote writes; it does not require remote
atomic support.

`--ipc-fork-test` keeps an owner allocation alive in the parent process,
exports a Level Zero IPC handle, launches a clean child process to open that
handle on a source device, fills it from the source device, then verifies from
the owner device. This is the closest small test to the vLLM TP worker setup.
