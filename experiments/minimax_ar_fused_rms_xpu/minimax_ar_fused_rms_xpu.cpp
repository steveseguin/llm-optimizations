#include <sycl/sycl.hpp>

#include <ATen/DeviceGuard.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <torch/csrc/distributed/c10d/Functional.hpp>
#include <torch/extension.h>
#include <torch/library.h>

#include <tuple>

#define CHECK_XPU(x) TORCH_CHECK((x).is_xpu(), #x " must be on XPU")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

namespace {

template <typename T>
struct SyclTypeTrait {
  using Type = T;
};

template <>
struct SyclTypeTrait<c10::Half> {
  using Type = sycl::half;
};

template <>
struct SyclTypeTrait<c10::BFloat16> {
  using Type = sycl::ext::oneapi::bfloat16;
};

template <typename scalar_t>
class MinimaxArFusedAddRmsKernel {
 public:
  MinimaxArFusedAddRmsKernel(const scalar_t* input,
                             const scalar_t* residual,
                             const scalar_t* weight,
                             scalar_t* out,
                             scalar_t* residual_out,
                             int hidden_size,
                             float eps)
      : input_(input),
        residual_(residual),
        weight_(weight),
        out_(out),
        residual_out_(residual_out),
        hidden_size_(hidden_size),
        eps_(eps) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<1>& item) const {
    const int row = item.get_group(0);
    const int local_id = item.get_local_id(0);
    const int local_size = item.get_local_range(0);
    const int row_offset = row * hidden_size_;

    float sumsq = 0.0f;
    for (int col = local_id; col < hidden_size_; col += local_size) {
      const int idx = row_offset + col;
      const float x =
          static_cast<float>(input_[idx]) + static_cast<float>(residual_[idx]);
      sumsq += x * x;
    }

    sumsq =
        sycl::reduce_over_group(item.get_group(), sumsq, sycl::plus<float>());
    const float scale =
        sycl::rsqrt(sumsq / static_cast<float>(hidden_size_) + eps_);

    for (int col = local_id; col < hidden_size_; col += local_size) {
      const int idx = row_offset + col;
      const float x =
          static_cast<float>(input_[idx]) + static_cast<float>(residual_[idx]);
      residual_out_[idx] = static_cast<scalar_t>(x);

      const scalar_t norm_cast = static_cast<scalar_t>(x * scale);
      const float weighted =
          static_cast<float>(norm_cast) * static_cast<float>(weight_[col]);
      out_[idx] = static_cast<scalar_t>(weighted);
    }
  }

 private:
  const scalar_t* __restrict__ input_;
  const scalar_t* __restrict__ residual_;
  const scalar_t* __restrict__ weight_;
  scalar_t* __restrict__ out_;
  scalar_t* __restrict__ residual_out_;
  const int hidden_size_;
  const float eps_;
};

template <typename scalar_t>
class MinimaxRmsKernel {
 public:
  MinimaxRmsKernel(const scalar_t* input,
                   const scalar_t* weight,
                   scalar_t* out,
                   int hidden_size,
                   float eps)
      : input_(input),
        weight_(weight),
        out_(out),
        hidden_size_(hidden_size),
        eps_(eps) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<1>& item) const {
    const int row = item.get_group(0);
    const int local_id = item.get_local_id(0);
    const int local_size = item.get_local_range(0);
    const int row_offset = row * hidden_size_;

    float sumsq = 0.0f;
    for (int col = local_id; col < hidden_size_; col += local_size) {
      const int idx = row_offset + col;
      const float x = static_cast<float>(input_[idx]);
      sumsq += x * x;
    }

    sumsq =
        sycl::reduce_over_group(item.get_group(), sumsq, sycl::plus<float>());
    const float scale =
        sycl::rsqrt(sumsq / static_cast<float>(hidden_size_) + eps_);

    for (int col = local_id; col < hidden_size_; col += local_size) {
      const int idx = row_offset + col;
      const float x = static_cast<float>(input_[idx]);
      const scalar_t norm_cast = static_cast<scalar_t>(x * scale);
      const float weighted =
          static_cast<float>(norm_cast) * static_cast<float>(weight_[col]);
      out_[idx] = static_cast<scalar_t>(weighted);
    }
  }

 private:
  const scalar_t* __restrict__ input_;
  const scalar_t* __restrict__ weight_;
  scalar_t* __restrict__ out_;
  const int hidden_size_;
  const float eps_;
};

template <typename scalar_t>
void launch_ar_fused_add_rms(const torch::Tensor& input,
                             const torch::Tensor& residual,
                             const torch::Tensor& weight,
                             torch::Tensor& out,
                             torch::Tensor& residual_out,
                             float eps) {
  using sycl_t = typename SyclTypeTrait<scalar_t>::Type;
  const int rows = input.size(0);
  const int hidden_size = input.size(1);
  constexpr int block_size = 256;

  auto& queue = c10::xpu::getCurrentXPUStream(input.device().index()).queue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(rows * block_size, block_size),
        MinimaxArFusedAddRmsKernel<sycl_t>(
            reinterpret_cast<const sycl_t*>(input.data_ptr<scalar_t>()),
            reinterpret_cast<const sycl_t*>(residual.data_ptr<scalar_t>()),
            reinterpret_cast<const sycl_t*>(weight.data_ptr<scalar_t>()),
            reinterpret_cast<sycl_t*>(out.data_ptr<scalar_t>()),
            reinterpret_cast<sycl_t*>(residual_out.data_ptr<scalar_t>()),
            hidden_size,
            eps));
  });
}

template <typename scalar_t>
void launch_rms(const torch::Tensor& input,
                const torch::Tensor& weight,
                torch::Tensor& out,
                float eps) {
  using sycl_t = typename SyclTypeTrait<scalar_t>::Type;
  const int rows = input.size(0);
  const int hidden_size = input.size(1);
  constexpr int block_size = 256;

  auto& queue = c10::xpu::getCurrentXPUStream(input.device().index()).queue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(rows * block_size, block_size),
        MinimaxRmsKernel<sycl_t>(
            reinterpret_cast<const sycl_t*>(input.data_ptr<scalar_t>()),
            reinterpret_cast<const sycl_t*>(weight.data_ptr<scalar_t>()),
            reinterpret_cast<sycl_t*>(out.data_ptr<scalar_t>()),
            hidden_size,
            eps));
  });
}

std::tuple<at::Tensor, at::Tensor> ar_fused_add_rms(
    const at::Tensor& input,
    const at::Tensor& residual,
    const at::Tensor& weight,
    const std::string& group_name,
    double eps) {
  const at::DeviceGuard device_guard(input.device());
  CHECK_XPU(input);
  CHECK_XPU(residual);
  CHECK_XPU(weight);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(residual);
  CHECK_CONTIGUOUS(weight);
  TORCH_CHECK(input.dim() == 2, "input must be rank-2");
  TORCH_CHECK(residual.sizes() == input.sizes(), "residual shape mismatch");
  TORCH_CHECK(weight.dim() == 1, "weight must be rank-1");
  TORCH_CHECK(weight.size(0) == input.size(1), "weight hidden size mismatch");
  TORCH_CHECK(input.scalar_type() == residual.scalar_type(),
              "input and residual dtype mismatch");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
              "input and weight dtype mismatch");

  at::Tensor reduced = c10d::all_reduce(input, "sum", group_name);
  reduced = c10d::wait_tensor(reduced);
  CHECK_XPU(reduced);
  CHECK_CONTIGUOUS(reduced);

  at::Tensor out = at::empty_like(input);
  at::Tensor residual_out = at::empty_like(input);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      input.scalar_type(), "minimax_ar_fused_add_rms_xpu", [&] {
        launch_ar_fused_add_rms<scalar_t>(reduced,
                                          residual,
                                          weight,
                                          out,
                                          residual_out,
                                          static_cast<float>(eps));
      });
  return std::make_tuple(out, residual_out);
}

std::tuple<at::Tensor, at::Tensor> ar_rms(const at::Tensor& input,
                                          const at::Tensor& weight,
                                          const std::string& group_name,
                                          double eps) {
  const at::DeviceGuard device_guard(input.device());
  CHECK_XPU(input);
  CHECK_XPU(weight);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);
  TORCH_CHECK(input.dim() == 2, "input must be rank-2");
  TORCH_CHECK(weight.dim() == 1, "weight must be rank-1");
  TORCH_CHECK(weight.size(0) == input.size(1), "weight hidden size mismatch");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
              "input and weight dtype mismatch");

  at::Tensor reduced = c10d::all_reduce(input, "sum", group_name);
  reduced = c10d::wait_tensor(reduced);
  CHECK_XPU(reduced);
  CHECK_CONTIGUOUS(reduced);

  at::Tensor out = at::empty_like(input);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      input.scalar_type(), "minimax_ar_rms_xpu", [&] {
        launch_rms<scalar_t>(
            reduced, weight, out, static_cast<float>(eps));
      });
  return std::make_tuple(out, reduced);
}

}  // namespace

TORCH_LIBRARY(minimax_ar_fused_rms_xpu, m) {
  m.def(
      "ar_fused_add_rms(Tensor input, Tensor residual, Tensor weight, str group_name, float eps) -> (Tensor, Tensor)");
  m.def(
      "ar_rms(Tensor input, Tensor weight, str group_name, float eps) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(minimax_ar_fused_rms_xpu, XPU, m) {
  m.impl("ar_fused_add_rms", ar_fused_add_rms);
  m.impl("ar_rms", ar_rms);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
