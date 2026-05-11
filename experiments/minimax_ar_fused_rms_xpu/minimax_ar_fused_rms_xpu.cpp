#include <torch/extension.h>
#include <torch/library.h>
#include <torch/csrc/distributed/c10d/Functional.hpp>

#include <tuple>

namespace {

std::tuple<at::Tensor, at::Tensor> ar_fused_add_rms(
    const at::Tensor& input,
    const at::Tensor& residual,
    const at::Tensor& weight,
    const std::string& group_name,
    double eps) {
  TORCH_CHECK(input.dim() == 2, "input must be rank-2");
  TORCH_CHECK(residual.sizes() == input.sizes(), "residual shape mismatch");
  TORCH_CHECK(weight.dim() == 1, "weight must be rank-1");
  TORCH_CHECK(weight.size(0) == input.size(1), "weight hidden size mismatch");

  at::Tensor reduced = c10d::all_reduce(input, "sum", group_name);
  reduced = c10d::wait_tensor(reduced);

  at::Tensor x = reduced.to(at::kFloat);
  at::Tensor residual_f = residual.scalar_type() == at::kFloat
      ? residual
      : residual.to(at::kFloat);
  x = x + residual_f;
  at::Tensor residual_out = x.to(input.scalar_type());

  at::Tensor variance = x.pow(2).mean(-1, true);
  at::Tensor y = x * at::rsqrt(variance + eps);
  at::Tensor out = y.to(input.scalar_type()) * weight;
  return std::make_tuple(out, residual_out);
}

}  // namespace

TORCH_LIBRARY(minimax_ar_fused_rms_xpu, m) {
  m.def(
      "ar_fused_add_rms(Tensor input, Tensor residual, Tensor weight, str group_name, float eps) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(minimax_ar_fused_rms_xpu, CompositeExplicitAutograd, m) {
  m.impl("ar_fused_add_rms", ar_fused_add_rms);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

