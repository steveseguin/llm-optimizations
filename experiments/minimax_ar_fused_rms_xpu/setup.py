import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, SyclExtension


sycl_targets = os.environ.get("MINIMAX_AR_FUSED_RMS_XPU_SYCL_TARGETS", "spir64")
sycl_flags = ["-fsycl", f"-fsycl-targets={sycl_targets}"]
if sycl_device := os.environ.get("MINIMAX_AR_FUSED_RMS_XPU_SYCL_DEVICE"):
    sycl_flags.extend(["-Xs", f"-device {sycl_device}"])

setup(
    name="minimax_ar_fused_rms_xpu",
    version="0.0.2",
    ext_modules=[
        SyclExtension(
            name="_minimax_ar_fused_rms_xpu",
            sources=["minimax_ar_fused_rms_xpu.cpp"],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++20",
                    "-fPIC",
                    *sycl_flags,
                ],
            },
            extra_link_args=sycl_flags,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
