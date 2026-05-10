#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#include <ATen/DeviceGuard.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <torch/extension.h>

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#define CHECK_XPU(x) TORCH_CHECK((x).is_xpu(), #x " must be on XPU")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

namespace {

void check_ze(ze_result_t result, const char* what) {
  TORCH_CHECK(result == ZE_RESULT_SUCCESS, what, " failed with ze_result=", result);
}

std::string bytes_to_hex(const void* data, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  std::ostringstream os;
  os << std::hex << std::setfill('0');
  for (size_t i = 0; i < size; ++i) {
    os << std::setw(2) << static_cast<unsigned>(bytes[i]);
  }
  return os.str();
}

bool hex_to_bytes(const std::string& hex, void* out, size_t size) {
  if (hex.size() != size * 2) {
    return false;
  }
  auto* bytes = static_cast<uint8_t*>(out);
  for (size_t i = 0; i < size; ++i) {
    char buf[3] = {hex[i * 2], hex[i * 2 + 1], '\0'};
    char* end = nullptr;
    unsigned long value = std::strtoul(buf, &end, 16);
    if (end == nullptr || *end != '\0' || value > 255) {
      return false;
    }
    bytes[i] = static_cast<uint8_t>(value);
  }
  return true;
}

ze_context_handle_t current_l0_context() {
  auto sycl_context = c10::xpu::get_device_context();
  return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_context);
}

ze_device_handle_t l0_device_for_index(int64_t device_index) {
  c10::xpu::check_device_index(static_cast<int>(device_index));
  auto sycl_device = c10::xpu::get_raw_device(static_cast<int>(device_index));
  return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);
}

class IpcQkVarAllreduceKernel {
 public:
  IpcQkVarAllreduceKernel(float* qk_var,
                          const int64_t* peer_ptrs,
                          int64_t num_items,
                          int64_t mailbox_offset,
                          int world_size,
                          int timeout_iters)
      : qk_var_(qk_var),
        peer_ptrs_(peer_ptrs),
        num_items_(num_items),
        mailbox_offset_(mailbox_offset),
        world_size_(world_size),
        timeout_iters_(timeout_iters) {}

  void operator()(sycl::id<1> id) const {
    const int64_t idx = static_cast<int64_t>(id[0]);
    if (idx >= num_items_) {
      return;
    }

    float* local = reinterpret_cast<float*>(peer_ptrs_[0]);
    local[mailbox_offset_ + idx] = qk_var_[idx];
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);

    float sum = 0.0f;
    for (int rank = 0; rank < world_size_; ++rank) {
      const volatile float* peer =
          reinterpret_cast<const volatile float*>(peer_ptrs_[rank]);
      float value = -0.0f;
      int iters = 0;
      while (iters < timeout_iters_) {
        sycl::atomic_fence(sycl::memory_order::seq_cst,
                           sycl::memory_scope::system);
        value = peer[mailbox_offset_ + idx];
        if (value != 0.0f || !sycl::signbit(value)) {
          break;
        }
        ++iters;
      }
      sum += value;
    }

    qk_var_[idx] = sum / static_cast<float>(world_size_);
  }

 private:
  float* qk_var_;
  const int64_t* peer_ptrs_;
  int64_t num_items_;
  int64_t mailbox_offset_;
  int world_size_;
  int timeout_iters_;
};

class IpcQkVarReduceKernel {
 public:
  IpcQkVarReduceKernel(float* qk_var,
                       const int64_t* peer_ptrs,
                       int64_t num_items,
                       int64_t mailbox_offset,
                       int world_size)
      : qk_var_(qk_var),
        peer_ptrs_(peer_ptrs),
        num_items_(num_items),
        mailbox_offset_(mailbox_offset),
        world_size_(world_size) {}

  void operator()(sycl::id<1> id) const {
    const int64_t idx = static_cast<int64_t>(id[0]);
    if (idx >= num_items_) {
      return;
    }

    float sum = 0.0f;
    for (int rank = 0; rank < world_size_; ++rank) {
      const float* peer = reinterpret_cast<const float*>(peer_ptrs_[rank]);
      sum += peer[mailbox_offset_ + idx];
    }
    qk_var_[idx] = sum / static_cast<float>(world_size_);
  }

 private:
  float* qk_var_;
  const int64_t* peer_ptrs_;
  int64_t num_items_;
  int64_t mailbox_offset_;
  int world_size_;
};

class IpcQkVarAllreduceSeqKernel {
 public:
  IpcQkVarAllreduceSeqKernel(float* qk_var,
                             const int64_t* payload_ptrs,
                             const int64_t* seq_ptrs,
                             int64_t num_items,
                             int64_t mailbox_offset,
                             int world_size,
                             int sequence,
                             int timeout_iters)
      : qk_var_(qk_var),
        payload_ptrs_(payload_ptrs),
        seq_ptrs_(seq_ptrs),
        num_items_(num_items),
        mailbox_offset_(mailbox_offset),
        world_size_(world_size),
        sequence_(sequence),
        timeout_iters_(timeout_iters) {}

  void operator()(sycl::id<1> id) const {
    const int64_t idx = static_cast<int64_t>(id[0]);
    if (idx >= num_items_) {
      return;
    }

    float* local_payload = reinterpret_cast<float*>(payload_ptrs_[0]);
    int32_t* local_seq = reinterpret_cast<int32_t*>(seq_ptrs_[0]);
    local_payload[mailbox_offset_ + idx] = qk_var_[idx];
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
    local_seq[mailbox_offset_ + idx] = sequence_;
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);

    float sum = 0.0f;
    for (int rank = 0; rank < world_size_; ++rank) {
      const volatile int32_t* peer_seq =
          reinterpret_cast<const volatile int32_t*>(seq_ptrs_[rank]);
      const volatile float* peer_payload =
          reinterpret_cast<const volatile float*>(payload_ptrs_[rank]);
      int seen = 0;
      int iters = 0;
      while (iters < timeout_iters_) {
        sycl::atomic_fence(sycl::memory_order::seq_cst,
                           sycl::memory_scope::system);
        seen = peer_seq[mailbox_offset_ + idx];
        if (seen == sequence_) {
          break;
        }
        ++iters;
      }
      sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
      sum += peer_payload[mailbox_offset_ + idx];
    }

    qk_var_[idx] = sum / static_cast<float>(world_size_);
  }

 private:
  float* qk_var_;
  const int64_t* payload_ptrs_;
  const int64_t* seq_ptrs_;
  int64_t num_items_;
  int64_t mailbox_offset_;
  int world_size_;
  int sequence_;
  int timeout_iters_;
};

class IpcQkVarAllreduceCounterKernel {
 public:
  IpcQkVarAllreduceCounterKernel(float* qk_var,
                                 const int64_t* payload_ptrs,
                                 const int64_t* seq_ptrs,
                                 const int32_t* counter,
                                 int64_t num_items,
                                 int64_t mailbox_offset,
                                 int64_t counter_slot,
                                 int world_size,
                                 int timeout_iters)
      : qk_var_(qk_var),
        payload_ptrs_(payload_ptrs),
        seq_ptrs_(seq_ptrs),
        counter_(counter),
        num_items_(num_items),
        mailbox_offset_(mailbox_offset),
        counter_slot_(counter_slot),
        world_size_(world_size),
        timeout_iters_(timeout_iters) {}

  void operator()(sycl::id<1> id) const {
    const int64_t idx = static_cast<int64_t>(id[0]);
    if (idx >= num_items_) {
      return;
    }

    const int sequence = counter_[counter_slot_];
    float* local_payload = reinterpret_cast<float*>(payload_ptrs_[0]);
    int32_t* local_seq = reinterpret_cast<int32_t*>(seq_ptrs_[0]);
    local_payload[mailbox_offset_ + idx] = qk_var_[idx];
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
    local_seq[mailbox_offset_ + idx] = sequence;
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);

    float sum = 0.0f;
    for (int rank = 0; rank < world_size_; ++rank) {
      const volatile int32_t* peer_seq =
          reinterpret_cast<const volatile int32_t*>(seq_ptrs_[rank]);
      const volatile float* peer_payload =
          reinterpret_cast<const volatile float*>(payload_ptrs_[rank]);
      int seen = 0;
      int iters = 0;
      while (iters < timeout_iters_) {
        sycl::atomic_fence(sycl::memory_order::seq_cst,
                           sycl::memory_scope::system);
        seen = peer_seq[mailbox_offset_ + idx];
        if (seen == sequence) {
          break;
        }
        ++iters;
      }
      sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
      sum += peer_payload[mailbox_offset_ + idx];
    }

    qk_var_[idx] = sum / static_cast<float>(world_size_);
  }

 private:
  float* qk_var_;
  const int64_t* payload_ptrs_;
  const int64_t* seq_ptrs_;
  const int32_t* counter_;
  int64_t num_items_;
  int64_t mailbox_offset_;
  int64_t counter_slot_;
  int world_size_;
  int timeout_iters_;
};

class IpcQkVarWritePayloadKernel {
 public:
  IpcQkVarWritePayloadKernel(const float* qk_var,
                             const int64_t* payload_ptrs,
                             int64_t num_items,
                             int64_t mailbox_offset)
      : qk_var_(qk_var),
        payload_ptrs_(payload_ptrs),
        num_items_(num_items),
        mailbox_offset_(mailbox_offset) {}

  void operator()(sycl::id<1> id) const {
    const int64_t idx = static_cast<int64_t>(id[0]);
    if (idx >= num_items_) {
      return;
    }
    float* local_payload = reinterpret_cast<float*>(payload_ptrs_[0]);
    local_payload[mailbox_offset_ + idx] = qk_var_[idx];
  }

 private:
  const float* qk_var_;
  const int64_t* payload_ptrs_;
  int64_t num_items_;
  int64_t mailbox_offset_;
};

class IpcQkVarReduceScalarSeqKernel {
 public:
  IpcQkVarReduceScalarSeqKernel(float* qk_var,
                                const int64_t* payload_ptrs,
                                const int64_t* seq_ptrs,
                                const int32_t* counter,
                                int64_t num_items,
                                int64_t mailbox_offset,
                                int64_t counter_slot,
                                int world_size,
                                int timeout_iters)
      : qk_var_(qk_var),
        payload_ptrs_(payload_ptrs),
        seq_ptrs_(seq_ptrs),
        counter_(counter),
        num_items_(num_items),
        mailbox_offset_(mailbox_offset),
        counter_slot_(counter_slot),
        world_size_(world_size),
        timeout_iters_(timeout_iters) {}

  void operator()(sycl::id<1> id) const {
    const int64_t idx = static_cast<int64_t>(id[0]);
    if (idx >= num_items_) {
      return;
    }

    const int sequence = counter_[counter_slot_];
    float sum = 0.0f;
    for (int rank = 0; rank < world_size_; ++rank) {
      const volatile int32_t* peer_seq =
          reinterpret_cast<const volatile int32_t*>(seq_ptrs_[rank]);
      const volatile float* peer_payload =
          reinterpret_cast<const volatile float*>(payload_ptrs_[rank]);
      int seen = 0;
      int iters = 0;
      while (iters < timeout_iters_) {
        sycl::atomic_fence(sycl::memory_order::seq_cst,
                           sycl::memory_scope::system);
        seen = peer_seq[counter_slot_];
        if (seen == sequence) {
          break;
        }
        ++iters;
      }
      sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
      sum += peer_payload[mailbox_offset_ + idx];
    }

    qk_var_[idx] = sum / static_cast<float>(world_size_);
  }

 private:
  float* qk_var_;
  const int64_t* payload_ptrs_;
  const int64_t* seq_ptrs_;
  const int32_t* counter_;
  int64_t num_items_;
  int64_t mailbox_offset_;
  int64_t counter_slot_;
  int world_size_;
  int timeout_iters_;
};

}  // namespace

std::string get_ipc_handle(torch::Tensor tensor) {
  const at::DeviceGuard device_guard(tensor.device());
  CHECK_XPU(tensor);
  CHECK_CONTIGUOUS(tensor);

  ze_ipc_mem_handle_t handle = {};
  check_ze(zeMemGetIpcHandle(current_l0_context(), tensor.data_ptr(), &handle),
           "zeMemGetIpcHandle");
  return bytes_to_hex(&handle, sizeof(handle));
}

uint64_t open_ipc_handle(const std::string& handle_hex, int64_t device_index) {
  c10::xpu::set_device(static_cast<int>(device_index));
  ze_ipc_mem_handle_t handle = {};
  TORCH_CHECK(hex_to_bytes(handle_hex, &handle, sizeof(handle)),
              "invalid IPC handle hex");

  void* ptr = nullptr;
  check_ze(zeMemOpenIpcHandle(
               current_l0_context(), l0_device_for_index(device_index), handle, 0, &ptr),
           "zeMemOpenIpcHandle");
  return reinterpret_cast<uint64_t>(ptr);
}

void close_ipc_handle(uint64_t ptr, int64_t device_index) {
  c10::xpu::set_device(static_cast<int>(device_index));
  check_ze(zeMemCloseIpcHandle(current_l0_context(),
                               reinterpret_cast<void*>(ptr)),
           "zeMemCloseIpcHandle");
}

void allreduce_qk_var(torch::Tensor qk_var,
                      torch::Tensor peer_ptrs,
                      int64_t slot,
                      int64_t max_tokens,
                      int64_t world_size,
                      int64_t timeout_iters) {
  const at::DeviceGuard device_guard(qk_var.device());
  CHECK_XPU(qk_var);
  CHECK_CONTIGUOUS(qk_var);
  CHECK_XPU(peer_ptrs);
  CHECK_CONTIGUOUS(peer_ptrs);
  TORCH_CHECK(qk_var.scalar_type() == torch::kFloat32, "qk_var must be float32");
  TORCH_CHECK(peer_ptrs.scalar_type() == torch::kInt64,
              "peer_ptrs must be int64");
  TORCH_CHECK(qk_var.dim() == 2 && qk_var.size(1) == 2,
              "qk_var must have shape [tokens, 2]");
  TORCH_CHECK(peer_ptrs.numel() >= world_size,
              "peer_ptrs must contain one pointer per rank");
  TORCH_CHECK(slot >= 0, "slot must be non-negative");
  TORCH_CHECK(max_tokens >= qk_var.size(0), "max_tokens too small");

  const int64_t num_items = qk_var.numel();
  const int64_t mailbox_offset = slot * max_tokens * 2;
  auto& queue = c10::xpu::getCurrentXPUStream(qk_var.device().index()).queue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::range<1>(num_items),
        IpcQkVarAllreduceKernel(qk_var.data_ptr<float>(),
                                peer_ptrs.data_ptr<int64_t>(),
                                num_items,
                                mailbox_offset,
                                static_cast<int>(world_size),
                                static_cast<int>(timeout_iters)));
  });
}

void reduce_qk_var_from_mailboxes(torch::Tensor qk_var,
                                  torch::Tensor peer_ptrs,
                                  int64_t slot,
                                  int64_t max_tokens,
                                  int64_t world_size) {
  const at::DeviceGuard device_guard(qk_var.device());
  CHECK_XPU(qk_var);
  CHECK_CONTIGUOUS(qk_var);
  CHECK_XPU(peer_ptrs);
  CHECK_CONTIGUOUS(peer_ptrs);
  TORCH_CHECK(qk_var.scalar_type() == torch::kFloat32, "qk_var must be float32");
  TORCH_CHECK(peer_ptrs.scalar_type() == torch::kInt64,
              "peer_ptrs must be int64");
  TORCH_CHECK(qk_var.dim() == 2 && qk_var.size(1) == 2,
              "qk_var must have shape [tokens, 2]");
  TORCH_CHECK(peer_ptrs.numel() >= world_size,
              "peer_ptrs must contain one pointer per rank");
  TORCH_CHECK(slot >= 0, "slot must be non-negative");
  TORCH_CHECK(max_tokens >= qk_var.size(0), "max_tokens too small");

  const int64_t num_items = qk_var.numel();
  const int64_t mailbox_offset = slot * max_tokens * 2;
  auto& queue = c10::xpu::getCurrentXPUStream(qk_var.device().index()).queue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::range<1>(num_items),
        IpcQkVarReduceKernel(qk_var.data_ptr<float>(),
                             peer_ptrs.data_ptr<int64_t>(),
                             num_items,
                             mailbox_offset,
                             static_cast<int>(world_size)));
  });
}

void allreduce_qk_var_seq(torch::Tensor qk_var,
                          torch::Tensor payload_ptrs,
                          torch::Tensor seq_ptrs,
                          int64_t slot,
                          int64_t sequence,
                          int64_t max_tokens,
                          int64_t world_size,
                          int64_t timeout_iters) {
  const at::DeviceGuard device_guard(qk_var.device());
  CHECK_XPU(qk_var);
  CHECK_CONTIGUOUS(qk_var);
  CHECK_XPU(payload_ptrs);
  CHECK_CONTIGUOUS(payload_ptrs);
  CHECK_XPU(seq_ptrs);
  CHECK_CONTIGUOUS(seq_ptrs);
  TORCH_CHECK(qk_var.scalar_type() == torch::kFloat32, "qk_var must be float32");
  TORCH_CHECK(payload_ptrs.scalar_type() == torch::kInt64,
              "payload_ptrs must be int64");
  TORCH_CHECK(seq_ptrs.scalar_type() == torch::kInt64, "seq_ptrs must be int64");
  TORCH_CHECK(qk_var.dim() == 2 && qk_var.size(1) == 2,
              "qk_var must have shape [tokens, 2]");
  TORCH_CHECK(payload_ptrs.numel() >= world_size,
              "payload_ptrs must contain one pointer per rank");
  TORCH_CHECK(seq_ptrs.numel() >= world_size,
              "seq_ptrs must contain one pointer per rank");
  TORCH_CHECK(slot >= 0, "slot must be non-negative");
  TORCH_CHECK(sequence > 0, "sequence must be positive");
  TORCH_CHECK(max_tokens >= qk_var.size(0), "max_tokens too small");

  const int64_t num_items = qk_var.numel();
  const int64_t mailbox_offset = slot * max_tokens * 2;
  auto& queue = c10::xpu::getCurrentXPUStream(qk_var.device().index()).queue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::range<1>(num_items),
        IpcQkVarAllreduceSeqKernel(qk_var.data_ptr<float>(),
                                   payload_ptrs.data_ptr<int64_t>(),
                                   seq_ptrs.data_ptr<int64_t>(),
                                   num_items,
                                   mailbox_offset,
                                   static_cast<int>(world_size),
                                   static_cast<int>(sequence),
                                   static_cast<int>(timeout_iters)));
  });
}

void allreduce_qk_var_seq_counter(torch::Tensor qk_var,
                                  torch::Tensor payload_ptrs,
                                  torch::Tensor seq_ptrs,
                                  torch::Tensor counter,
                                  int64_t slot,
                                  int64_t max_tokens,
                                  int64_t world_size,
                                  int64_t timeout_iters) {
  const at::DeviceGuard device_guard(qk_var.device());
  CHECK_XPU(qk_var);
  CHECK_CONTIGUOUS(qk_var);
  CHECK_XPU(payload_ptrs);
  CHECK_CONTIGUOUS(payload_ptrs);
  CHECK_XPU(seq_ptrs);
  CHECK_CONTIGUOUS(seq_ptrs);
  CHECK_XPU(counter);
  CHECK_CONTIGUOUS(counter);
  TORCH_CHECK(qk_var.scalar_type() == torch::kFloat32, "qk_var must be float32");
  TORCH_CHECK(payload_ptrs.scalar_type() == torch::kInt64,
              "payload_ptrs must be int64");
  TORCH_CHECK(seq_ptrs.scalar_type() == torch::kInt64, "seq_ptrs must be int64");
  TORCH_CHECK(counter.scalar_type() == torch::kInt32, "counter must be int32");
  TORCH_CHECK(qk_var.dim() == 2 && qk_var.size(1) == 2,
              "qk_var must have shape [tokens, 2]");
  TORCH_CHECK(payload_ptrs.numel() >= world_size,
              "payload_ptrs must contain one pointer per rank");
  TORCH_CHECK(seq_ptrs.numel() >= world_size,
              "seq_ptrs must contain one pointer per rank");
  TORCH_CHECK(slot >= 0, "slot must be non-negative");
  TORCH_CHECK(slot < counter.numel(), "counter slot out of range");
  TORCH_CHECK(max_tokens >= qk_var.size(0), "max_tokens too small");

  const int64_t num_items = qk_var.numel();
  const int64_t mailbox_offset = slot * max_tokens * 2;
  auto& queue = c10::xpu::getCurrentXPUStream(qk_var.device().index()).queue();
  int32_t* counter_ptr = counter.data_ptr<int32_t>();
  auto counter_event = queue.submit([&](sycl::handler& cgh) {
    cgh.single_task([=]() {
      counter_ptr[slot] += 1;
      sycl::atomic_fence(sycl::memory_order::seq_cst,
                         sycl::memory_scope::system);
    });
  });
  queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(counter_event);
    cgh.parallel_for(
        sycl::range<1>(num_items),
        IpcQkVarAllreduceCounterKernel(qk_var.data_ptr<float>(),
                                       payload_ptrs.data_ptr<int64_t>(),
                                       seq_ptrs.data_ptr<int64_t>(),
                                       counter.data_ptr<int32_t>(),
                                       num_items,
                                       mailbox_offset,
                                       slot,
                                       static_cast<int>(world_size),
                                       static_cast<int>(timeout_iters)));
  });
}

void allreduce_qk_var_seq_scalar(torch::Tensor qk_var,
                                 torch::Tensor payload_ptrs,
                                 torch::Tensor seq_ptrs,
                                 torch::Tensor counter,
                                 int64_t slot,
                                 int64_t max_tokens,
                                 int64_t world_size,
                                 int64_t timeout_iters) {
  const at::DeviceGuard device_guard(qk_var.device());
  CHECK_XPU(qk_var);
  CHECK_CONTIGUOUS(qk_var);
  CHECK_XPU(payload_ptrs);
  CHECK_CONTIGUOUS(payload_ptrs);
  CHECK_XPU(seq_ptrs);
  CHECK_CONTIGUOUS(seq_ptrs);
  CHECK_XPU(counter);
  CHECK_CONTIGUOUS(counter);
  TORCH_CHECK(qk_var.scalar_type() == torch::kFloat32, "qk_var must be float32");
  TORCH_CHECK(payload_ptrs.scalar_type() == torch::kInt64,
              "payload_ptrs must be int64");
  TORCH_CHECK(seq_ptrs.scalar_type() == torch::kInt64, "seq_ptrs must be int64");
  TORCH_CHECK(counter.scalar_type() == torch::kInt32, "counter must be int32");
  TORCH_CHECK(qk_var.dim() == 2 && qk_var.size(1) == 2,
              "qk_var must have shape [tokens, 2]");
  TORCH_CHECK(payload_ptrs.numel() >= world_size,
              "payload_ptrs must contain one pointer per rank");
  TORCH_CHECK(seq_ptrs.numel() >= world_size,
              "seq_ptrs must contain one pointer per rank");
  TORCH_CHECK(slot >= 0, "slot must be non-negative");
  TORCH_CHECK(slot < counter.numel(), "counter slot out of range");
  TORCH_CHECK(max_tokens >= qk_var.size(0), "max_tokens too small");

  const int64_t num_items = qk_var.numel();
  const int64_t mailbox_offset = slot * max_tokens * 2;
  auto& queue = c10::xpu::getCurrentXPUStream(qk_var.device().index()).queue();
  int32_t* counter_ptr = counter.data_ptr<int32_t>();
  const int64_t* payload_ptrs_data = payload_ptrs.data_ptr<int64_t>();
  const int64_t* seq_ptrs_data = seq_ptrs.data_ptr<int64_t>();
  auto counter_event = queue.submit([&](sycl::handler& cgh) {
    cgh.single_task([=]() {
      counter_ptr[slot] += 1;
      sycl::atomic_fence(sycl::memory_order::seq_cst,
                         sycl::memory_scope::system);
    });
  });
  auto write_event = queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(counter_event);
    cgh.parallel_for(
        sycl::range<1>(num_items),
        IpcQkVarWritePayloadKernel(qk_var.data_ptr<float>(),
                                   payload_ptrs_data,
                                   num_items,
                                   mailbox_offset));
  });
  auto publish_event = queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(write_event);
    cgh.single_task([=]() {
      int32_t* local_seq = reinterpret_cast<int32_t*>(seq_ptrs_data[0]);
      sycl::atomic_fence(sycl::memory_order::seq_cst,
                         sycl::memory_scope::system);
      local_seq[slot] = counter_ptr[slot];
      sycl::atomic_fence(sycl::memory_order::seq_cst,
                         sycl::memory_scope::system);
    });
  });
  queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(publish_event);
    cgh.parallel_for(
        sycl::range<1>(num_items),
        IpcQkVarReduceScalarSeqKernel(qk_var.data_ptr<float>(),
                                      payload_ptrs.data_ptr<int64_t>(),
                                      seq_ptrs.data_ptr<int64_t>(),
                                      counter.data_ptr<int32_t>(),
                                      num_items,
                                      mailbox_offset,
                                      slot,
                                      static_cast<int>(world_size),
                                      static_cast<int>(timeout_iters)));
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_ipc_handle", &get_ipc_handle, "Export a tensor's Level Zero IPC handle");
  m.def("open_ipc_handle", &open_ipc_handle, "Open a Level Zero IPC handle");
  m.def("close_ipc_handle", &close_ipc_handle, "Close a Level Zero IPC pointer");
  m.def("allreduce_qk_var",
        &allreduce_qk_var,
        "In-place peer-memory allreduce average for [tokens, 2] qk_var");
  m.def("reduce_qk_var_from_mailboxes",
        &reduce_qk_var_from_mailboxes,
        "Reduce [tokens, 2] qk_var from already-written peer mailboxes");
  m.def("allreduce_qk_var_seq",
        &allreduce_qk_var_seq,
        "Single-kernel sequence-mailbox allreduce average for [tokens, 2] qk_var");
  m.def("allreduce_qk_var_seq_counter",
        &allreduce_qk_var_seq_counter,
        "Device-counter sequence-mailbox allreduce average for [tokens, 2] qk_var");
  m.def("allreduce_qk_var_seq_scalar",
        &allreduce_qk_var_seq_scalar,
        "Scalar-sequence mailbox allreduce average for [tokens, 2] qk_var");
}

TORCH_LIBRARY(minimax_qk_rms_xpu_ipc, m) {
  m.def("allreduce_qk_var(Tensor! qk_var, Tensor peer_ptrs, int slot, int max_tokens, int world_size, int timeout_iters) -> ()");
  m.def("reduce_qk_var_from_mailboxes(Tensor! qk_var, Tensor peer_ptrs, int slot, int max_tokens, int world_size) -> ()");
  m.def("allreduce_qk_var_seq(Tensor! qk_var, Tensor payload_ptrs, Tensor seq_ptrs, int slot, int sequence, int max_tokens, int world_size, int timeout_iters) -> ()");
  m.def("allreduce_qk_var_seq_counter(Tensor! qk_var, Tensor payload_ptrs, Tensor seq_ptrs, Tensor! counter, int slot, int max_tokens, int world_size, int timeout_iters) -> ()");
  m.def("allreduce_qk_var_seq_scalar(Tensor! qk_var, Tensor payload_ptrs, Tensor seq_ptrs, Tensor! counter, int slot, int max_tokens, int world_size, int timeout_iters) -> ()");
}

TORCH_LIBRARY_IMPL(minimax_qk_rms_xpu_ipc, XPU, m) {
  m.impl("allreduce_qk_var", &allreduce_qk_var);
  m.impl("reduce_qk_var_from_mailboxes", &reduce_qk_var_from_mailboxes);
  m.impl("allreduce_qk_var_seq", &allreduce_qk_var_seq);
  m.impl("allreduce_qk_var_seq_counter", &allreduce_qk_var_seq_counter);
  m.impl("allreduce_qk_var_seq_scalar", &allreduce_qk_var_seq_scalar);
}
