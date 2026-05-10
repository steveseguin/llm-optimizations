# B70 PCIe and xpu-smi Check - 2026-05-10

## Summary

After the four-B70 reboot, the Linux DRM/sysfs endpoint fields still reported every B70 as `2.5 GT/s PCIe x1`. That is not the external slot link. Intel documents that Arc cards expose an internal PCIe hierarchy where normal OS tools can show Gen1 x1 on the GPU function even when the card is operating normally.

Root/slot-facing link checks show all four installed B70s running at PCIe 5.0 x16:

| xpu-smi id | GPU BDF | host root port | card upstream bridge | internal downstream endpoint |
| --- | --- | --- | --- | --- |
| 0 | `0000:03:00.0` | `0000:00:01.1`: 32GT/s x16 | `0000:01:00.0`: 32GT/s x16 | `0000:02:01.0` / GPU endpoint: 2.5GT/s x1 |
| 1 | `0000:83:00.0` | `0000:80:01.1`: 32GT/s x16 | `0000:81:00.0`: 32GT/s x16 | `0000:82:01.0` / GPU endpoint: 2.5GT/s x1 |
| 2 | `0000:a3:00.0` | `0000:a0:01.1`: 32GT/s x16 | `0000:a1:00.0`: 32GT/s x16 | `0000:a2:01.0` / GPU endpoint: 2.5GT/s x1 |
| 3 | `0000:e3:00.0` | `0000:e0:01.1`: 32GT/s x16 | `0000:e1:00.0`: 32GT/s x16 | `0000:e2:01.0` / GPU endpoint: 2.5GT/s x1 |

`xpu-smi config -d <id> -j` reports `pcie_downgrade_current_state: "disabled"` for all four cards when run as root. No power limits or frequency ranges were changed.

## Tooling Installed

`xpu-smi` was installed from the Canonical/Intel `ppa:kobuk-team/intel-graphics` source referenced by Intel's Ubuntu 24.04 client GPU installation guide.

The machine already had Intel's oneAPI `level-zero` package installed at `1.28.2`, which owns `/usr/lib/x86_64-linux-gnu/libze_loader.so.1`. The PPA's `xpu-smi` package depends on a package named `libze1`; installing the PPA `libze1` would overwrite the existing loader with `1.28.0`. To avoid replacing the working Level Zero loader, a local no-file package named `libze1` version `1.28.2-local1` was installed and held. It depends on `level-zero (>= 1.28.2)` and only satisfies the package-name dependency for `xpu-smi` and `libxpum1`.

Current relevant package state:

- `xpu-smi`: `1.3.6-1~24.04~ppa1`
- `libxpum1`: `1.3.6-1~24.04~ppa1`
- `level-zero`: `1.28.2`
- `libze1`: `1.28.2-local1`, held, no files
- `intel-opencl-icd`: `26.14.37833.4-0`
- `libze-intel-gpu1`: `26.14.37833.4-0`

## Notes

- `xpu-smi discovery -d <id> -j` still returns `pcie_generation: "-1"` and `pcie_max_link_width: "-1"` for these B70s, so it is not currently a reliable PCIe generation reporter on this setup.
- `xpu-smi topology -m` reports all four GPUs as `NODE` peers with CPU affinity `0-15`; the host is a single-socket 8-core/16-thread EPYC 9015, so there is no cross-NUMA CPU placement issue.
- `xpu-smi stats -e` provides power/frequency/memory-used data on B70 but EU utilization and memory bandwidth are still `N/A`.
- The PCIe downgrade knob exists in this `xpu-smi` build as `--pciedowngrade`, but it is already disabled. Do not use it unless a future boot shows the setting enabled or the slot-facing bridge drops below 32GT/s x16.

## References

- Intel Arc PCIe hierarchy note: https://www.intel.com/content/www/us/en/support/articles/000094587/graphics.html
- Intel client GPU Ubuntu 24.04 install guide: https://dgpu-docs.intel.com/driver/client/overview.html
- XPU-SMI installation guide: https://intel.github.io/xpumanager/smi_install_guide.html
