#include <sycl/sycl.hpp>

#include <ATen/DeviceGuard.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <torch/extension.h>

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
class MinimaxQkRmsVarKernel {
 public:
  MinimaxQkRmsVarKernel(const scalar_t* qkv,
                        float* qk_var,
                        int q_size,
                        int kv_size,
                        int qkv_stride)
      : qkv_(qkv),
        qk_var_(qk_var),
        q_size_(q_size),
        kv_size_(kv_size),
        qkv_stride_(qkv_stride) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<1>& item) const {
    const int group = item.get_group(0);
    const int token_idx = group >> 1;
    const int part = group & 1;
    const int hidden = part == 0 ? q_size_ : kv_size_;
    const int offset = token_idx * qkv_stride_ + (part == 0 ? 0 : q_size_);

    float sum = 0.0f;
    for (int i = item.get_local_id(0); i < hidden;
         i += item.get_local_range(0)) {
      float x = static_cast<float>(qkv_[offset + i]);
      sum += x * x;
    }

    sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>());
    if (item.get_local_id(0) == 0) {
      qk_var_[token_idx * 2 + part] = sum / static_cast<float>(hidden);
    }
  }

 private:
  const scalar_t* __restrict__ qkv_;
  float* __restrict__ qk_var_;
  const int q_size_;
  const int kv_size_;
  const int qkv_stride_;
};

template <typename scalar_t>
class MinimaxQkRmsApplyKernel {
 public:
  MinimaxQkRmsApplyKernel(const scalar_t* qkv,
                          const float* qk_var,
                          const scalar_t* q_weight,
                          const scalar_t* k_weight,
                          scalar_t* q_out,
                          scalar_t* k_out,
                          int q_size,
                          int kv_size,
                          int qkv_stride,
                          float eps)
      : qkv_(qkv),
        qk_var_(qk_var),
        q_weight_(q_weight),
        k_weight_(k_weight),
        q_out_(q_out),
        k_out_(k_out),
        q_size_(q_size),
        kv_size_(kv_size),
        qkv_stride_(qkv_stride),
        eps_(eps) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<1>& item) const {
    const int group = item.get_group(0);
    const int token_idx = group >> 1;
    const int part = group & 1;
    const int hidden = part == 0 ? q_size_ : kv_size_;
    const int input_offset =
        token_idx * qkv_stride_ + (part == 0 ? 0 : q_size_);
    const int output_offset = token_idx * hidden;
    const float scale = sycl::rsqrt(qk_var_[token_idx * 2 + part] + eps_);

    for (int i = item.get_local_id(0); i < hidden;
         i += item.get_local_range(0)) {
      float x = static_cast<float>(qkv_[input_offset + i]);
      float w = static_cast<float>((part == 0 ? q_weight_ : k_weight_)[i]);
      scalar_t y = static_cast<scalar_t>(x * scale * w);
      if (part == 0) {
        q_out_[output_offset + i] = y;
      } else {
        k_out_[output_offset + i] = y;
      }
    }
  }

 private:
  const scalar_t* __restrict__ qkv_;
  const float* __restrict__ qk_var_;
  const scalar_t* __restrict__ q_weight_;
  const scalar_t* __restrict__ k_weight_;
  scalar_t* __restrict__ q_out_;
  scalar_t* __restrict__ k_out_;
  const int q_size_;
  const int kv_size_;
  const int qkv_stride_;
  const float eps_;
};

template <typename scalar_t>
class MinimaxQkRmsApplyQkRopeKernel {
 public:
  MinimaxQkRmsApplyQkRopeKernel(const float* q,
                                const float* k,
                                const float* qk_var,
                                const scalar_t* q_weight,
                                const scalar_t* k_weight,
                                const int64_t* positions,
                                const scalar_t* cos_sin_cache,
                                scalar_t* q_out,
                                scalar_t* k_out,
                                int q_size,
                                int kv_size,
                                int head_size,
                                int rotary_dim,
                                float eps,
                                bool is_neox)
      : q_(q),
        k_(k),
        qk_var_(qk_var),
        q_weight_(q_weight),
        k_weight_(k_weight),
        positions_(positions),
        cos_sin_cache_(cos_sin_cache),
        q_out_(q_out),
        k_out_(k_out),
        q_size_(q_size),
        kv_size_(kv_size),
        head_size_(head_size),
        rotary_dim_(rotary_dim),
        eps_(eps),
        is_neox_(is_neox) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<1>& item) const {
    const int group = item.get_group(0);
    const int token_idx = group >> 1;
    const int part = group & 1;
    const int hidden = part == 0 ? q_size_ : kv_size_;
    const float* input = part == 0 ? q_ : k_;
    const scalar_t* weight = part == 0 ? q_weight_ : k_weight_;
    scalar_t* output = part == 0 ? q_out_ : k_out_;
    const float scale = sycl::rsqrt(qk_var_[token_idx * 2 + part] + eps_);
    const int64_t pos = positions_[token_idx];
    const scalar_t* cache = cos_sin_cache_ + pos * rotary_dim_;
    const int embed_dim = rotary_dim_ / 2;

    for (int i = item.get_local_id(0); i < hidden;
         i += item.get_local_range(0)) {
      const int token_offset = token_idx * hidden;
      const int dim = i % head_size_;
      float out = scaled(input, weight, token_offset, i, scale);

      if (dim < rotary_dim_) {
        const int head_offset = i - dim;
        int x_dim;
        int y_dim;
        int rot_offset;
        bool is_x;
        if (is_neox_) {
          is_x = dim < embed_dim;
          rot_offset = is_x ? dim : dim - embed_dim;
          x_dim = head_offset + rot_offset;
          y_dim = head_offset + embed_dim + rot_offset;
        } else {
          is_x = (dim % 2) == 0;
          rot_offset = dim / 2;
          x_dim = head_offset + rot_offset * 2;
          y_dim = x_dim + 1;
        }
        const float x = scaled(input, weight, token_offset, x_dim, scale);
        const float y = scaled(input, weight, token_offset, y_dim, scale);
        const float cos_v = static_cast<float>(cache[rot_offset]);
        const float sin_v = static_cast<float>(cache[embed_dim + rot_offset]);
        out = is_x ? (x * cos_v - y * sin_v) : (y * cos_v + x * sin_v);
      }

      output[token_offset + i] = static_cast<scalar_t>(out);
    }
  }

 private:
  float scaled(const float* input,
               const scalar_t* weight,
               int token_offset,
               int idx,
               float scale) const {
    return input[token_offset + idx] * scale * static_cast<float>(weight[idx]);
  }

  const float* __restrict__ q_;
  const float* __restrict__ k_;
  const float* __restrict__ qk_var_;
  const scalar_t* __restrict__ q_weight_;
  const scalar_t* __restrict__ k_weight_;
  const int64_t* __restrict__ positions_;
  const scalar_t* __restrict__ cos_sin_cache_;
  scalar_t* __restrict__ q_out_;
  scalar_t* __restrict__ k_out_;
  const int q_size_;
  const int kv_size_;
  const int head_size_;
  const int rotary_dim_;
  const float eps_;
  const bool is_neox_;
};

template <typename scalar_t>
void launch_var(torch::Tensor& qkv,
                torch::Tensor& qk_var,
                int q_size,
                int kv_size) {
  using sycl_t = typename SyclTypeTrait<scalar_t>::Type;
  const int num_tokens = qkv.size(0);
  const int qkv_stride = qkv.size(1);
  constexpr int block_size = 256;
  auto& queue = c10::xpu::getCurrentXPUStream(qkv.device().index()).queue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(num_tokens * 2 * block_size, block_size),
        MinimaxQkRmsVarKernel<sycl_t>(
            reinterpret_cast<const sycl_t*>(qkv.data_ptr<scalar_t>()),
            qk_var.data_ptr<float>(),
            q_size,
            kv_size,
            qkv_stride));
  });
}

template <typename scalar_t>
void launch_apply(torch::Tensor& qkv,
                  torch::Tensor& qk_var,
                  torch::Tensor& q_weight,
                  torch::Tensor& k_weight,
                  torch::Tensor& q_out,
                  torch::Tensor& k_out,
                  int q_size,
                  int kv_size,
                  float eps) {
  using sycl_t = typename SyclTypeTrait<scalar_t>::Type;
  const int num_tokens = qkv.size(0);
  const int qkv_stride = qkv.size(1);
  constexpr int block_size = 256;
  auto& queue = c10::xpu::getCurrentXPUStream(qkv.device().index()).queue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(num_tokens * 2 * block_size, block_size),
        MinimaxQkRmsApplyKernel<sycl_t>(
            reinterpret_cast<const sycl_t*>(qkv.data_ptr<scalar_t>()),
            qk_var.data_ptr<float>(),
            reinterpret_cast<const sycl_t*>(q_weight.data_ptr<scalar_t>()),
            reinterpret_cast<const sycl_t*>(k_weight.data_ptr<scalar_t>()),
            reinterpret_cast<sycl_t*>(q_out.data_ptr<scalar_t>()),
            reinterpret_cast<sycl_t*>(k_out.data_ptr<scalar_t>()),
            q_size,
            kv_size,
            qkv_stride,
            eps));
      });
}

template <typename scalar_t>
void launch_apply_qk_rope(torch::Tensor& q,
                          torch::Tensor& k,
                          torch::Tensor& qk_var,
                          torch::Tensor& q_weight,
                          torch::Tensor& k_weight,
                          torch::Tensor& positions,
                          torch::Tensor& cos_sin_cache,
                          torch::Tensor& q_out,
                          torch::Tensor& k_out,
                          int head_size,
                          int rotary_dim,
                          float eps,
                          bool is_neox) {
  using sycl_t = typename SyclTypeTrait<scalar_t>::Type;
  const int num_tokens = q.size(0);
  const int q_size = q.size(1);
  const int kv_size = k.size(1);
  constexpr int block_size = 256;
  auto& queue = c10::xpu::getCurrentXPUStream(q_out.device().index()).queue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(num_tokens * 2 * block_size, block_size),
        MinimaxQkRmsApplyQkRopeKernel<sycl_t>(
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            qk_var.data_ptr<float>(),
            reinterpret_cast<const sycl_t*>(q_weight.data_ptr<scalar_t>()),
            reinterpret_cast<const sycl_t*>(k_weight.data_ptr<scalar_t>()),
            positions.data_ptr<int64_t>(),
            reinterpret_cast<const sycl_t*>(
                cos_sin_cache.data_ptr<scalar_t>()),
            reinterpret_cast<sycl_t*>(q_out.data_ptr<scalar_t>()),
            reinterpret_cast<sycl_t*>(k_out.data_ptr<scalar_t>()),
            q_size,
            kv_size,
            head_size,
            rotary_dim,
            eps,
            is_neox));
  });
}

}  // namespace

void var(torch::Tensor qkv,
         torch::Tensor qk_var,
         int64_t q_size,
         int64_t kv_size) {
  const at::DeviceGuard device_guard(qkv.device());
  CHECK_XPU(qkv);
  CHECK_CONTIGUOUS(qkv);
  CHECK_XPU(qk_var);
  CHECK_CONTIGUOUS(qk_var);
  TORCH_CHECK(qkv.dim() == 2, "qkv must be 2D");
  TORCH_CHECK(qk_var.dim() == 2, "qk_var must be 2D");
  TORCH_CHECK(qk_var.scalar_type() == torch::kFloat32, "qk_var must be float32");
  TORCH_CHECK(qk_var.size(0) == qkv.size(0), "qk_var token dim must match qkv");
  TORCH_CHECK(qk_var.size(1) == 2, "qk_var must have shape [num_tokens, 2]");
  TORCH_CHECK(qkv.size(1) == q_size + 2 * kv_size, "qkv hidden dim mismatch");

  AT_DISPATCH_REDUCED_FLOATING_TYPES(qkv.scalar_type(), "minimax_qk_rms_var", [&] {
    launch_var<scalar_t>(
        qkv, qk_var, static_cast<int>(q_size), static_cast<int>(kv_size));
  });
}

void apply(torch::Tensor qkv,
           torch::Tensor qk_var,
           torch::Tensor q_weight,
           torch::Tensor k_weight,
           torch::Tensor q_out,
           torch::Tensor k_out,
           int64_t q_size,
           int64_t kv_size,
           double eps) {
  const at::DeviceGuard device_guard(qkv.device());
  CHECK_XPU(qkv);
  CHECK_CONTIGUOUS(qkv);
  CHECK_XPU(qk_var);
  CHECK_CONTIGUOUS(qk_var);
  CHECK_XPU(q_weight);
  CHECK_CONTIGUOUS(q_weight);
  CHECK_XPU(k_weight);
  CHECK_CONTIGUOUS(k_weight);
  CHECK_XPU(q_out);
  CHECK_CONTIGUOUS(q_out);
  CHECK_XPU(k_out);
  CHECK_CONTIGUOUS(k_out);
  TORCH_CHECK(qkv.dim() == 2, "qkv must be 2D");
  TORCH_CHECK(qk_var.dim() == 2, "qk_var must be 2D");
  TORCH_CHECK(qk_var.scalar_type() == torch::kFloat32, "qk_var must be float32");
  TORCH_CHECK(qk_var.size(0) == qkv.size(0), "qk_var token dim must match qkv");
  TORCH_CHECK(qk_var.size(1) == 2, "qk_var must have shape [num_tokens, 2]");
  TORCH_CHECK(qkv.size(1) == q_size + 2 * kv_size, "qkv hidden dim mismatch");
  TORCH_CHECK(q_weight.size(0) == q_size, "q_weight size mismatch");
  TORCH_CHECK(k_weight.size(0) == kv_size, "k_weight size mismatch");
  TORCH_CHECK(q_out.size(0) == qkv.size(0) && q_out.size(1) == q_size,
              "q_out shape mismatch");
  TORCH_CHECK(k_out.size(0) == qkv.size(0) && k_out.size(1) == kv_size,
              "k_out shape mismatch");
  TORCH_CHECK(qkv.scalar_type() == q_weight.scalar_type() &&
                  qkv.scalar_type() == k_weight.scalar_type() &&
                  qkv.scalar_type() == q_out.scalar_type() &&
                  qkv.scalar_type() == k_out.scalar_type(),
              "qkv, weights, and outputs must have the same dtype");

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      qkv.scalar_type(), "minimax_qk_rms_apply", [&] {
        launch_apply<scalar_t>(qkv,
                               qk_var,
                               q_weight,
                               k_weight,
                               q_out,
                               k_out,
                               static_cast<int>(q_size),
                               static_cast<int>(kv_size),
                               static_cast<float>(eps));
      });
}

void apply_qk_rope(torch::Tensor q,
                   torch::Tensor k,
                   torch::Tensor qk_var,
                   torch::Tensor q_weight,
                   torch::Tensor k_weight,
                   torch::Tensor positions,
                   torch::Tensor cos_sin_cache,
                   torch::Tensor q_out,
                   torch::Tensor k_out,
                   int64_t head_size,
                   int64_t rotary_dim,
                   double eps,
                   bool is_neox) {
  const at::DeviceGuard device_guard(q_out.device());
  CHECK_XPU(q);
  CHECK_CONTIGUOUS(q);
  CHECK_XPU(k);
  CHECK_CONTIGUOUS(k);
  CHECK_XPU(qk_var);
  CHECK_CONTIGUOUS(qk_var);
  CHECK_XPU(q_weight);
  CHECK_CONTIGUOUS(q_weight);
  CHECK_XPU(k_weight);
  CHECK_CONTIGUOUS(k_weight);
  CHECK_XPU(positions);
  CHECK_CONTIGUOUS(positions);
  CHECK_XPU(cos_sin_cache);
  CHECK_CONTIGUOUS(cos_sin_cache);
  CHECK_XPU(q_out);
  CHECK_CONTIGUOUS(q_out);
  CHECK_XPU(k_out);
  CHECK_CONTIGUOUS(k_out);
  TORCH_CHECK(q.dim() == 2, "q must be 2D");
  TORCH_CHECK(k.dim() == 2, "k must be 2D");
  TORCH_CHECK(q.scalar_type() == torch::kFloat32, "q must be float32");
  TORCH_CHECK(k.scalar_type() == torch::kFloat32, "k must be float32");
  TORCH_CHECK(qk_var.dim() == 2, "qk_var must be 2D");
  TORCH_CHECK(qk_var.scalar_type() == torch::kFloat32, "qk_var must be float32");
  TORCH_CHECK(qk_var.size(0) == q.size(0), "qk_var token dim must match q");
  TORCH_CHECK(qk_var.size(1) == 2, "qk_var must have shape [num_tokens, 2]");
  TORCH_CHECK(k.size(0) == q.size(0), "q and k token dims must match");
  TORCH_CHECK(positions.dim() == 1, "positions must be 1D");
  TORCH_CHECK(positions.scalar_type() == torch::kInt64, "positions must be int64");
  TORCH_CHECK(positions.size(0) == q.size(0), "positions token dim mismatch");
  TORCH_CHECK(q_out.size(0) == q.size(0) && q_out.size(1) == q.size(1),
              "q_out shape mismatch");
  TORCH_CHECK(k_out.size(0) == k.size(0) && k_out.size(1) == k.size(1),
              "k_out shape mismatch");
  TORCH_CHECK(q_weight.size(0) == q.size(1), "q_weight size mismatch");
  TORCH_CHECK(k_weight.size(0) == k.size(1), "k_weight size mismatch");
  TORCH_CHECK(q_out.scalar_type() == k_out.scalar_type() &&
                  q_out.scalar_type() == q_weight.scalar_type() &&
                  q_out.scalar_type() == k_weight.scalar_type() &&
                  q_out.scalar_type() == cos_sin_cache.scalar_type(),
              "outputs, weights, and cos_sin_cache must have the same dtype");
  TORCH_CHECK(q.size(1) % head_size == 0, "q hidden dim must divide head_size");
  TORCH_CHECK(k.size(1) % head_size == 0, "k hidden dim must divide head_size");
  TORCH_CHECK(rotary_dim > 0 && rotary_dim <= head_size && rotary_dim % 2 == 0,
              "rotary_dim must be positive, even, and <= head_size");
  TORCH_CHECK(cos_sin_cache.dim() == 2, "cos_sin_cache must be 2D");
  TORCH_CHECK(cos_sin_cache.size(1) == rotary_dim,
              "cos_sin_cache rotary dim mismatch");

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      q_out.scalar_type(), "minimax_qk_rms_apply_qk_rope", [&] {
        launch_apply_qk_rope<scalar_t>(q,
                                       k,
                                       qk_var,
                                       q_weight,
                                       k_weight,
                                       positions,
                                       cos_sin_cache,
                                       q_out,
                                       k_out,
                                       static_cast<int>(head_size),
                                       static_cast<int>(rotary_dim),
                                       static_cast<float>(eps),
                                       is_neox);
      });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("var", &var, "MiniMax Q/K RMS variance helper");
  m.def("apply", &apply, "MiniMax Q/K RMS apply helper");
  m.def("apply_qk_rope",
        &apply_qk_rope,
        "MiniMax Q/K RMS apply plus RoPE helper");
}

TORCH_LIBRARY(minimax_qk_rms_xpu, m) {
  m.def("var(Tensor qkv, Tensor! qk_var, int q_size, int kv_size) -> ()");
  m.def(
      "apply(Tensor qkv, Tensor qk_var, Tensor q_weight, Tensor k_weight, "
      "Tensor! q_out, Tensor! k_out, int q_size, int kv_size, float eps) -> ()");
  m.def(
      "apply_qk_rope(Tensor q, Tensor k, Tensor qk_var, Tensor q_weight, "
      "Tensor k_weight, Tensor positions, Tensor cos_sin_cache, Tensor! q_out, "
      "Tensor! k_out, int head_size, int rotary_dim, float eps, bool is_neox) "
      "-> ()");
}

TORCH_LIBRARY_IMPL(minimax_qk_rms_xpu, XPU, m) {
  m.impl("var", &var);
  m.impl("apply", &apply);
  m.impl("apply_qk_rope", &apply_qk_rope);
}
