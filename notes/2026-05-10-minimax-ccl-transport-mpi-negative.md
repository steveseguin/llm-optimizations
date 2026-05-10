# MiniMax CCL MPI Transport Negative

## Summary

Tried replacing the known-working oneCCL OFI transport with MPI:

```text
CCL_ATL_TRANSPORT=mpi
USE_LLM_SCALER_MOE=1
XPU_GRAPH=0
INPUT_LEN=512 OUTPUT_LEN=256 MAX_MODEL_LEN=2048 MAX_BATCHED_TOKENS=1024
MAX_NUM_SEQS=1 NUM_PROMPTS=1 TP=4
```

The run failed before model load. No benchmark JSON was produced.

Log:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260510T001548Z.log
```

## Failure Signature

The worker stack fails in MPI-backed oneCCL initialization:

```text
MPIDI_GPU_init_mpl_global
MPIDI_GPU_init
PMPI_Init_thread
atl_mpi::init
atl_mpi_comm::init_transport
atl_mpi_comm::atl_mpi_comm
WorkerProc initialization failed due to an exception in a background process
RuntimeError: Engine core initialization failed
```

It exits in about 33 seconds, before model load or KV allocation.

## Interpretation

`CCL_ATL_TRANSPORT=mpi` is not a viable path for this current vLLM/XPU 4x B70
setup. Keep `CCL_ATL_TRANSPORT=ofi`, which is the known-working transport for
the MiniMax TP=4 runs.

This is a useful negative result for reproduction notes, but it should not be
submitted as a LocalMaxxing benchmark because inference never started.
