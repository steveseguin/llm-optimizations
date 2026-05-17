#include <sycl/sycl.hpp>

#include <ATen/DeviceGuard.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <torch/csrc/distributed/c10d/Functional.hpp>
#include <torch/extension.h>

#include <cfloat>
#include <cstdint>

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
class FillPairKernel {
 public:
  FillPairKernel(const scalar_t* values,
                 const int64_t* tokens,
                 float* pair,
                 int64_t batch)
      : values_(values), tokens_(tokens), pair_(pair), batch_(batch) {}

  void operator()(const sycl::id<1>& id) const {
    const int64_t i = static_cast<int64_t>(id[0]);
    if (i >= batch_) {
      return;
    }
    pair_[i * 2] = static_cast<float>(values_[i]);
    pair_[i * 2 + 1] = static_cast<float>(tokens_[i]);
  }

 private:
  const scalar_t* __restrict__ values_;
  const int64_t* __restrict__ tokens_;
  float* __restrict__ pair_;
  const int64_t batch_;
};

class ReducePairKernel {
 public:
  ReducePairKernel(const float* gathered,
                   int64_t* out_tokens,
                   int64_t batch,
                   int64_t world_size)
      : gathered_(gathered),
        out_tokens_(out_tokens),
        batch_(batch),
        world_size_(world_size) {}

  void operator()(const sycl::id<1>& id) const {
    const int64_t b = static_cast<int64_t>(id[0]);
    if (b >= batch_) {
      return;
    }

    uint32_t best_key = ordered_key(gathered_[b * 2]);
    float best_token = gathered_[b * 2 + 1];
    for (int64_t rank = 1; rank < world_size_; ++rank) {
      const int64_t offset = (rank * batch_ + b) * 2;
      const float value = gathered_[offset];
      const float token = gathered_[offset + 1];
      const uint32_t key = ordered_key(value);
      if (key > best_key) {
        best_key = key;
        best_token = token;
      }
    }
    out_tokens_[b] = static_cast<int64_t>(best_token);
  }

 private:
  static uint32_t ordered_key(float value) {
    const uint32_t bits = sycl::bit_cast<uint32_t>(value);
    return (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);
  }

  const float* __restrict__ gathered_;
  int64_t* __restrict__ out_tokens_;
  const int64_t batch_;
  const int64_t world_size_;
};

class ReduceFlatPairKernel {
 public:
  ReduceFlatPairKernel(const float* gathered,
                       int64_t* out_tokens,
                       int64_t batch,
                       int64_t world_size)
      : gathered_(gathered),
        out_tokens_(out_tokens),
        batch_(batch),
        world_size_(world_size) {}

  void operator()(const sycl::id<1>& id) const {
    const int64_t b = static_cast<int64_t>(id[0]);
    if (b >= batch_) {
      return;
    }

    const int64_t row_offset = b * world_size_ * 2;
    uint32_t best_key = ordered_key(gathered_[row_offset]);
    float best_token = gathered_[row_offset + 1];
    for (int64_t rank = 1; rank < world_size_; ++rank) {
      const int64_t offset = row_offset + rank * 2;
      const uint32_t key = ordered_key(gathered_[offset]);
      if (key > best_key) {
        best_key = key;
        best_token = gathered_[offset + 1];
      }
    }
    out_tokens_[b] = static_cast<int64_t>(best_token);
  }

 private:
  static uint32_t ordered_key(float value) {
    const uint32_t bits = sycl::bit_cast<uint32_t>(value);
    return (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);
  }

  const float* __restrict__ gathered_;
  int64_t* __restrict__ out_tokens_;
  const int64_t batch_;
  const int64_t world_size_;
};

template <typename scalar_t>
void launch_fill_pair(const at::Tensor& values,
                      const at::Tensor& tokens,
                      at::Tensor& pair) {
  using sycl_t = typename SyclTypeTrait<scalar_t>::Type;
  const int64_t batch = values.numel();
  auto& queue = c10::xpu::getCurrentXPUStream(values.device().index()).queue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::range<1>(batch),
        FillPairKernel<sycl_t>(
            reinterpret_cast<const sycl_t*>(values.data_ptr<scalar_t>()),
            tokens.data_ptr<int64_t>(),
            pair.data_ptr<float>(),
            batch));
  });
}

void launch_reduce_pair(const at::Tensor& gathered,
                        at::Tensor& out_tokens,
                        int64_t batch,
                        int64_t world_size) {
  auto& queue =
      c10::xpu::getCurrentXPUStream(gathered.device().index()).queue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::range<1>(batch),
        ReducePairKernel(
            gathered.data_ptr<float>(),
            out_tokens.data_ptr<int64_t>(),
            batch,
            world_size));
  });
}

void launch_reduce_flat_pair(const at::Tensor& gathered,
                             at::Tensor& out_tokens,
                             int64_t batch,
                             int64_t world_size) {
  auto& queue =
      c10::xpu::getCurrentXPUStream(gathered.device().index()).queue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::range<1>(batch),
        ReduceFlatPairKernel(
            gathered.data_ptr<float>(),
            out_tokens.data_ptr<int64_t>(),
            batch,
            world_size));
  });
}

at::Tensor pair_argmax(const at::Tensor& values,
                       const at::Tensor& tokens,
                       const std::string& group_name,
                       int64_t world_size) {
  const at::DeviceGuard device_guard(values.device());
  CHECK_XPU(values);
  CHECK_XPU(tokens);
  CHECK_CONTIGUOUS(values);
  CHECK_CONTIGUOUS(tokens);
  TORCH_CHECK(values.dim() == 1, "values must be rank-1");
  TORCH_CHECK(tokens.dim() == 1, "tokens must be rank-1");
  TORCH_CHECK(tokens.scalar_type() == at::kLong, "tokens must be int64");
  TORCH_CHECK(values.numel() == tokens.numel(), "values/tokens shape mismatch");
  TORCH_CHECK(world_size > 0, "world_size must be positive");
  TORCH_CHECK(values.scalar_type() == at::kHalf ||
                  values.scalar_type() == at::kBFloat16 ||
                  values.scalar_type() == at::kFloat,
              "values must be float16, bfloat16, or float32");

  const int64_t batch = values.numel();
  auto pair = at::empty({batch, 2}, values.options().dtype(at::kFloat));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      values.scalar_type(),
      "minimax_pair_argmax_fill_xpu",
      [&] { launch_fill_pair<scalar_t>(values, tokens, pair); });
  // c10d functional collectives do not inherit the SYCL kernel dependency from
  // the current queue on this stack. Fence the tiny fill before gathering.
  c10::xpu::getCurrentXPUStream(values.device().index()).synchronize();

  at::Tensor gathered = c10d::all_gather_into_tensor(pair, world_size, group_name);
  gathered = c10d::wait_tensor(gathered);
  CHECK_XPU(gathered);
  CHECK_CONTIGUOUS(gathered);

  auto out_tokens = at::empty({batch}, tokens.options());
  launch_reduce_pair(gathered, out_tokens, batch, world_size);
  return out_tokens;
}

at::Tensor reduce_flat_pairs(const at::Tensor& gathered, int64_t world_size) {
  const at::DeviceGuard device_guard(gathered.device());
  CHECK_XPU(gathered);
  CHECK_CONTIGUOUS(gathered);
  TORCH_CHECK(gathered.dim() == 2, "gathered must be rank-2");
  TORCH_CHECK(gathered.scalar_type() == at::kFloat, "gathered must be float32");
  TORCH_CHECK(world_size > 0, "world_size must be positive");
  TORCH_CHECK(gathered.size(1) == world_size * 2,
              "gathered second dimension must be world_size * 2");

  const int64_t batch = gathered.size(0);
  auto out_tokens =
      at::empty({batch}, gathered.options().dtype(at::kLong));
  launch_reduce_flat_pair(gathered, out_tokens, batch, world_size);
  return out_tokens;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pair_argmax", &pair_argmax, "MiniMax local argmax pair reducer (XPU)");
  m.def("reduce_flat_pairs",
        &reduce_flat_pairs,
        "Reduce gathered [batch, 2 * world] MiniMax local-argmax pairs (XPU)");
}
