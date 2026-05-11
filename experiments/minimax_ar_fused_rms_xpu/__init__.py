from pathlib import Path

import torch


def _load_extension() -> None:
    here = Path(__file__).resolve().parent
    matches = sorted(here.glob("_minimax_ar_fused_rms_xpu*.so"))
    matches += sorted((here / "build").glob("_minimax_ar_fused_rms_xpu*.so"))
    if not matches:
        raise ImportError(
            "MiniMax allreduce fused RMS extension is not built. "
            "Run setup.py build_ext --inplace in this directory first."
        )
    torch.ops.load_library(str(matches[-1]))


_load_extension()


@torch.library.register_fake("minimax_ar_fused_rms_xpu::ar_fused_add_rms")
def _fake_ar_fused_add_rms(input, residual, weight, group_name: str, eps: float):
    return torch.empty_like(input), torch.empty_like(input)
