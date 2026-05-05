# 2026-05-05 xe Reset Deadlock Recovery Note

## Trigger

After MiniMax M2.7 SYCL experiments produced Level Zero `UR_RESULT_ERROR_DEVICE_LOST` and one forced-DMMV segmentation fault, Qwen multi-device runs started failing below llama.cpp:

- single-card Qwen still completed with f16 KV;
- 3-card Qwen aborted inside oneMKL GEMM with `UR_RESULT_ERROR_DEVICE_LOST`;
- a tiny SYCL peer-read test initially reproduced instability on all four devices.

## Attempted Recovery

The four B70 VGA functions were PCI-reset through sysfs:

```text
0000:03:00.0
0000:83:00.0
0000:a3:00.0
0000:e3:00.0
```

That did not cleanly recover the stack. `sycl-ls` and the peer-read test then aborted in NEO DRM initialization:

```text
Abort was called at 445 line in file:
../../neo/shared/source/os_interface/linux/drm_neo.cpp
```

An `xe` unbind/rebind was attempted next. Rebinding `0000:83:00.0` entered uninterruptible kernel sleep and could not be killed.

## Current State

Only one B70 is currently bound and visible to Level Zero:

- visible: `level_zero:0`, UUID/BDF `...0300...`, PCI `0000:03:00.0`;
- stuck during bind: `0000:83:00.0`;
- unbound after interrupted recovery sequence: `0000:a3:00.0`, `0000:e3:00.0`;
- a kernel D-state task remains from the `xe` bind path.

Kernel evidence:

```text
task kworker/0:5 blocked for more than 122 seconds
workqueue: events work_for_cpu_fn
intel_edp_init_connector
intel_dp_init_connector
intel_setup_outputs
xe_display_init_early
xe_device_probe
xe_pci_probe
```

This is a kernel-driver deadlock in the display-probe path, not a user-space benchmark bug.

## Recovery Recommendation

A reboot is required to clear the uninterruptible kernel task.

Before reboot, configure the `xe` driver to avoid display probing for these headless compute GPUs:

```text
options xe disable_display=1 probe_display=0
```

Rationale:

- the machine has an ASPEED display controller, so the B70 display outputs are not needed for console/display;
- the deadlock is in `xe_display_init_early` and connector probing;
- disabling display probing should leave render/compute nodes available while avoiding the fragile display-power path.

## After Reboot Validation

Run these checks before continuing benchmarks:

```bash
source /opt/intel/oneapi/setvars.sh --force
sycl-ls
ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3 /home/steve/sycl-peer-read-test
```

Then retry the Qwen Q4 known-good 3-card selector before any MiniMax or q8_0 experiments:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:2,1,3 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_SYCL_Q8_CACHE=1 \
GGML_SYCL_COMM_ALLREDUCE=1 \
GGML_SYCL_COMM_SINGLE_KERNEL=1 \
GGML_SYCL_COMM_EVENT_BARRIER=1 \
GGML_META_FUSE_ALLREDUCE_ADD=1 \
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench \
  -m /home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf \
  -dev SYCL0,SYCL1,SYCL2 -sm row -ts 1/1/1 \
  -ngl 99 -fa 1 -ub 32 -ctk f16 -ctv f16 \
  -p 16 -n 8 -r 1 --no-warmup --poll 50 -o jsonl
```

## LocalMaxxing

Not submitted. This is a runtime recovery blocker, not a benchmark.
