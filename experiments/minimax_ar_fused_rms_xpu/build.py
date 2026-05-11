from pathlib import Path

from torch.utils.cpp_extension import load


HERE = Path(__file__).resolve().parent

load(
    name="_minimax_ar_fused_rms_xpu",
    sources=[str(HERE / "minimax_ar_fused_rms_xpu.cpp")],
    build_directory=str(HERE / "build"),
    extra_cflags=["-O3"],
    verbose=True,
)

