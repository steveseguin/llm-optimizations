#include <level_zero/ze_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

const char* result_name(ze_result_t r) {
  switch (r) {
    case ZE_RESULT_SUCCESS:
      return "ZE_RESULT_SUCCESS";
    case ZE_RESULT_NOT_READY:
      return "ZE_RESULT_NOT_READY";
    case ZE_RESULT_ERROR_UNINITIALIZED:
      return "ZE_RESULT_ERROR_UNINITIALIZED";
    case ZE_RESULT_ERROR_DEVICE_LOST:
      return "ZE_RESULT_ERROR_DEVICE_LOST";
    case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
      return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
    case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
      return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
    case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
      return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
    case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
      return "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
    case ZE_RESULT_ERROR_INVALID_SIZE:
      return "ZE_RESULT_ERROR_INVALID_SIZE";
    case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
      return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
    case ZE_RESULT_ERROR_INVALID_ARGUMENT:
      return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
    default:
      return "ZE_RESULT_OTHER";
  }
}

void check(ze_result_t r, const char* what) {
  if (r != ZE_RESULT_SUCCESS) {
    std::cerr << what << " failed: " << result_name(r) << " (" << r << ")\n";
    std::exit(1);
  }
}

int command_ordinal(ze_device_handle_t device) {
  uint32_t count = 0;
  check(zeDeviceGetCommandQueueGroupProperties(device, &count, nullptr),
        "zeDeviceGetCommandQueueGroupProperties(count)");
  std::vector<ze_command_queue_group_properties_t> props(count);
  check(zeDeviceGetCommandQueueGroupProperties(device, &count, props.data()),
        "zeDeviceGetCommandQueueGroupProperties(list)");
  for (uint32_t i = 0; i < count; ++i) {
    if (props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      return static_cast<int>(i);
    }
  }
  for (uint32_t i = 0; i < count; ++i) {
    if (props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

ze_result_t run_fill(ze_context_handle_t context,
                     ze_device_handle_t device,
                     uint32_t ordinal,
                     void* ptr,
                     size_t bytes,
                     uint32_t pattern) {
  ze_command_queue_desc_t queue_desc = {};
  queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  queue_desc.ordinal = ordinal;
  queue_desc.index = 0;
  queue_desc.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
  queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

  ze_command_queue_handle_t queue = nullptr;
  ze_result_t r = zeCommandQueueCreate(context, device, &queue_desc, &queue);
  if (r != ZE_RESULT_SUCCESS) {
    return r;
  }

  ze_command_list_desc_t list_desc = {};
  list_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
  list_desc.commandQueueGroupOrdinal = ordinal;
  ze_command_list_handle_t list = nullptr;
  r = zeCommandListCreate(context, device, &list_desc, &list);
  if (r == ZE_RESULT_SUCCESS) {
    r = zeCommandListAppendMemoryFill(
        list, ptr, &pattern, sizeof(pattern), bytes, nullptr, 0, nullptr);
  }
  if (r == ZE_RESULT_SUCCESS) {
    r = zeCommandListClose(list);
  }
  if (r == ZE_RESULT_SUCCESS) {
    r = zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr);
  }
  if (r == ZE_RESULT_SUCCESS) {
    r = zeCommandQueueSynchronize(queue, UINT64_MAX);
  }
  if (list) {
    zeCommandListDestroy(list);
  }
  zeCommandQueueDestroy(queue);
  return r;
}

ze_result_t run_copy_to_host(ze_context_handle_t context,
                             ze_device_handle_t device,
                             uint32_t ordinal,
                             void* dst_host,
                             const void* src_device,
                             size_t bytes) {
  ze_command_queue_desc_t queue_desc = {};
  queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  queue_desc.ordinal = ordinal;
  queue_desc.index = 0;
  queue_desc.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
  queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

  ze_command_queue_handle_t queue = nullptr;
  ze_result_t r = zeCommandQueueCreate(context, device, &queue_desc, &queue);
  if (r != ZE_RESULT_SUCCESS) {
    return r;
  }

  ze_command_list_desc_t list_desc = {};
  list_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
  list_desc.commandQueueGroupOrdinal = ordinal;
  ze_command_list_handle_t list = nullptr;
  r = zeCommandListCreate(context, device, &list_desc, &list);
  if (r == ZE_RESULT_SUCCESS) {
    r = zeCommandListAppendMemoryCopy(
        list, dst_host, src_device, bytes, nullptr, 0, nullptr);
  }
  if (r == ZE_RESULT_SUCCESS) {
    r = zeCommandListClose(list);
  }
  if (r == ZE_RESULT_SUCCESS) {
    r = zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr);
  }
  if (r == ZE_RESULT_SUCCESS) {
    r = zeCommandQueueSynchronize(queue, UINT64_MAX);
  }
  if (list) {
    zeCommandListDestroy(list);
  }
  zeCommandQueueDestroy(queue);
  return r;
}

void run_p2p_fill_test(ze_driver_handle_t driver,
                       const std::vector<ze_device_handle_t>& devices) {
  constexpr size_t bytes = 4096;
  constexpr uint32_t base_pattern = 0x5a170000u;

  ze_context_desc_t context_desc = {};
  context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  ze_context_handle_t context = nullptr;
  check(zeContextCreate(driver, &context_desc, &context), "zeContextCreate");

  std::vector<int> ordinals(devices.size(), -1);
  for (size_t i = 0; i < devices.size(); ++i) {
    ordinals[i] = command_ordinal(devices[i]);
    if (ordinals[i] < 0) {
      std::cerr << "device[" << i << "] has no compute/copy queue group\n";
      std::exit(1);
    }
  }

  std::cout << "p2p_fill_test bytes=" << bytes << "\n";
  for (size_t dst = 0; dst < devices.size(); ++dst) {
    ze_device_mem_alloc_desc_t alloc_desc = {};
    alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    alloc_desc.ordinal = 0;
    void* ptr = nullptr;
    ze_result_t r =
        zeMemAllocDevice(context, &alloc_desc, bytes, 64, devices[dst], &ptr);
    if (r != ZE_RESULT_SUCCESS) {
      std::cout << "fill[*->" << dst << "] alloc_result=" << result_name(r)
                << "\n";
      continue;
    }

    for (size_t src = 0; src < devices.size(); ++src) {
      uint32_t pattern = base_pattern | static_cast<uint32_t>((src << 4) | dst);
      r = run_fill(context,
                   devices[src],
                   static_cast<uint32_t>(ordinals[src]),
                   ptr,
                   bytes,
                   pattern);

      std::vector<uint32_t> host(bytes / sizeof(uint32_t), 0);
      ze_result_t copy_result = run_copy_to_host(context,
                                                devices[dst],
                                                static_cast<uint32_t>(ordinals[dst]),
                                                host.data(),
                                                ptr,
                                                bytes);
      bool ok = r == ZE_RESULT_SUCCESS && copy_result == ZE_RESULT_SUCCESS;
      for (uint32_t value : host) {
        ok = ok && value == pattern;
      }

      std::cout << "fill[" << src << "->" << dst << "]"
                << " fill_result=" << result_name(r)
                << " copy_result=" << result_name(copy_result)
                << " verify=" << (ok ? "ok" : "bad") << "\n";
    }

    zeMemFree(context, ptr);
  }

  zeContextDestroy(context);
}

std::string uuid_hex(const ze_device_uuid_t& uuid) {
  std::ostringstream os;
  os << std::hex << std::setfill('0');
  for (uint8_t b : uuid.id) {
    os << std::setw(2) << static_cast<unsigned>(b);
  }
  return os.str();
}

std::string external_flags(ze_external_memory_type_flags_t flags) {
  std::vector<const char*> names;
  if (flags & ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD) {
    names.push_back("OPAQUE_FD");
  }
  if (flags & ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF) {
    names.push_back("DMA_BUF");
  }
  if (flags & ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32) {
    names.push_back("OPAQUE_WIN32");
  }
  if (flags & ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32_KMT) {
    names.push_back("OPAQUE_WIN32_KMT");
  }
  if (flags & ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D11_TEXTURE) {
    names.push_back("D3D11_TEXTURE");
  }
  if (flags & ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D11_TEXTURE_KMT) {
    names.push_back("D3D11_TEXTURE_KMT");
  }
  if (flags & ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D12_HEAP) {
    names.push_back("D3D12_HEAP");
  }
  if (flags & ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D12_RESOURCE) {
    names.push_back("D3D12_RESOURCE");
  }
  if (names.empty()) {
    return "none";
  }
  std::ostringstream os;
  for (size_t i = 0; i < names.size(); ++i) {
    if (i) {
      os << "|";
    }
    os << names[i];
  }
  return os.str();
}

std::string p2p_flags(ze_device_p2p_property_flags_t flags) {
  std::vector<const char*> names;
  if (flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS) {
    names.push_back("ACCESS");
  }
  if (flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ATOMICS) {
    names.push_back("ATOMICS");
  }
  if (names.empty()) {
    return "none";
  }
  std::ostringstream os;
  for (size_t i = 0; i < names.size(); ++i) {
    if (i) {
      os << "|";
    }
    os << names[i];
  }
  return os.str();
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

std::vector<ze_driver_handle_t> get_drivers() {
  uint32_t driver_count = 0;
  check(zeDriverGet(&driver_count, nullptr), "zeDriverGet(count)");
  std::vector<ze_driver_handle_t> drivers(driver_count);
  check(zeDriverGet(&driver_count, drivers.data()), "zeDriverGet(list)");
  return drivers;
}

std::vector<ze_device_handle_t> get_devices(ze_driver_handle_t driver) {
  uint32_t device_count = 0;
  check(zeDeviceGet(driver, &device_count, nullptr), "zeDeviceGet(count)");
  std::vector<ze_device_handle_t> devices(device_count);
  check(zeDeviceGet(driver, &device_count, devices.data()), "zeDeviceGet(list)");
  return devices;
}

int ipc_child(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: " << argv[0]
              << " --ipc-child <src_idx> <handle_hex> <pattern_hex>\n";
    return 2;
  }

  const int src = std::atoi(argv[2]);
  const std::string handle_hex = argv[3];
  const uint32_t pattern =
      static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 16));
  constexpr size_t bytes = 4096;

  ze_ipc_mem_handle_t handle = {};
  if (!hex_to_bytes(handle_hex, &handle, sizeof(handle))) {
    std::cerr << "invalid IPC handle hex\n";
    return 2;
  }

  check(zeInit(0), "child zeInit");
  auto drivers = get_drivers();
  if (drivers.empty()) {
    std::cerr << "child found no Level Zero drivers\n";
    return 1;
  }
  auto devices = get_devices(drivers[0]);
  if (src < 0 || static_cast<size_t>(src) >= devices.size()) {
    std::cerr << "child source device index out of range\n";
    return 2;
  }

  ze_context_desc_t context_desc = {};
  context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  ze_context_handle_t context = nullptr;
  check(zeContextCreate(drivers[0], &context_desc, &context),
        "child zeContextCreate");

  void* peer_ptr = nullptr;
  ze_result_t open_result =
      zeMemOpenIpcHandle(context, devices[src], handle, 0, &peer_ptr);
  if (open_result != ZE_RESULT_SUCCESS) {
    std::cerr << "child zeMemOpenIpcHandle failed: "
              << result_name(open_result) << "\n";
    zeContextDestroy(context);
    return 1;
  }

  const int ordinal = command_ordinal(devices[src]);
  ze_result_t fill_result = run_fill(context,
                                     devices[src],
                                     static_cast<uint32_t>(ordinal),
                                     peer_ptr,
                                     bytes,
                                     pattern);
  if (fill_result != ZE_RESULT_SUCCESS) {
    std::cerr << "child fill failed: " << result_name(fill_result) << "\n";
  }

  zeMemCloseIpcHandle(context, peer_ptr);
  zeContextDestroy(context);
  return fill_result == ZE_RESULT_SUCCESS ? 0 : 1;
}

void run_ipc_fork_test(const char* self_path,
                       ze_driver_handle_t driver,
                       const std::vector<ze_device_handle_t>& devices) {
  constexpr size_t bytes = 4096;
  constexpr uint32_t base_pattern = 0x1dc00000u;

  ze_context_desc_t context_desc = {};
  context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  ze_context_handle_t context = nullptr;
  check(zeContextCreate(driver, &context_desc, &context), "zeContextCreate");

  std::vector<int> ordinals(devices.size(), -1);
  for (size_t i = 0; i < devices.size(); ++i) {
    ordinals[i] = command_ordinal(devices[i]);
    if (ordinals[i] < 0) {
      std::cerr << "device[" << i << "] has no compute/copy queue group\n";
      std::exit(1);
    }
  }

  std::cout << "ipc_fork_test bytes=" << bytes << "\n";
  for (size_t dst = 0; dst < devices.size(); ++dst) {
    ze_device_mem_alloc_desc_t alloc_desc = {};
    alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    alloc_desc.ordinal = 0;
    void* ptr = nullptr;
    ze_result_t r =
        zeMemAllocDevice(context, &alloc_desc, bytes, 64, devices[dst], &ptr);
    if (r != ZE_RESULT_SUCCESS) {
      std::cout << "ipc[*->" << dst << "] alloc_result=" << result_name(r)
                << "\n";
      continue;
    }

    ze_ipc_mem_handle_t handle = {};
    r = zeMemGetIpcHandle(context, ptr, &handle);
    if (r != ZE_RESULT_SUCCESS) {
      std::cout << "ipc[*->" << dst << "] get_handle_result=" << result_name(r)
                << "\n";
      zeMemFree(context, ptr);
      continue;
    }
    const std::string handle_hex = bytes_to_hex(&handle, sizeof(handle));

    for (size_t src = 0; src < devices.size(); ++src) {
      const uint32_t pattern =
          base_pattern | static_cast<uint32_t>((src << 4) | dst);
      std::ostringstream src_arg;
      std::ostringstream pattern_arg;
      src_arg << src;
      pattern_arg << std::hex << pattern;

      pid_t pid = fork();
      if (pid == 0) {
        execl(self_path,
              self_path,
              "--ipc-child",
              src_arg.str().c_str(),
              handle_hex.c_str(),
              pattern_arg.str().c_str(),
              static_cast<char*>(nullptr));
        std::perror("execl");
        _exit(127);
      }

      int status = 1;
      if (pid < 0 || waitpid(pid, &status, 0) < 0) {
        std::perror("fork/waitpid");
      }
      const bool child_ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;

      std::vector<uint32_t> host(bytes / sizeof(uint32_t), 0);
      const ze_result_t copy_result =
          run_copy_to_host(context,
                           devices[dst],
                           static_cast<uint32_t>(ordinals[dst]),
                           host.data(),
                           ptr,
                           bytes);
      bool verify_ok = child_ok && copy_result == ZE_RESULT_SUCCESS;
      for (uint32_t value : host) {
        verify_ok = verify_ok && value == pattern;
      }

      std::cout << "ipc[" << src << "->" << dst << "]"
                << " child=" << (child_ok ? "ok" : "bad")
                << " copy_result=" << result_name(copy_result)
                << " verify=" << (verify_ok ? "ok" : "bad") << "\n";
    }

    zeMemPutIpcHandle(context, handle);
    zeMemFree(context, ptr);
  }

  zeContextDestroy(context);
}

}  // namespace

int main(int argc, char** argv) {
  bool do_p2p_fill_test = false;
  bool do_ipc_fork_test = false;
  if (argc >= 2 && std::strcmp(argv[1], "--ipc-child") == 0) {
    return ipc_child(argc, argv);
  }
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--p2p-fill-test") == 0) {
      do_p2p_fill_test = true;
    } else if (std::strcmp(argv[i], "--ipc-fork-test") == 0) {
      do_ipc_fork_test = true;
    } else {
      std::cerr << "unknown argument: " << argv[i] << "\n";
      return 2;
    }
  }

  check(zeInit(0), "zeInit");

  std::vector<ze_driver_handle_t> drivers = get_drivers();
  uint32_t driver_count = static_cast<uint32_t>(drivers.size());

  std::cout << "drivers=" << driver_count << "\n";

  for (uint32_t d = 0; d < driver_count; ++d) {
    std::vector<ze_device_handle_t> devices = get_devices(drivers[d]);
    uint32_t device_count = static_cast<uint32_t>(devices.size());

    std::cout << "driver[" << d << "].devices=" << device_count << "\n";

    std::vector<ze_device_properties_t> props(device_count);
    for (uint32_t i = 0; i < device_count; ++i) {
      props[i] = {};
      props[i].stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      check(zeDeviceGetProperties(devices[i], &props[i]),
            "zeDeviceGetProperties");

      ze_device_external_memory_properties_t ext = {};
      ext.stype = ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES;
      check(zeDeviceGetExternalMemoryProperties(devices[i], &ext),
            "zeDeviceGetExternalMemoryProperties");

      std::cout << "device[" << i << "]"
                << " name=\"" << props[i].name << "\""
                << " vendor=0x" << std::hex << props[i].vendorId
                << " device=0x" << props[i].deviceId << std::dec
                << " subdevice="
                << ((props[i].flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE) ? "yes"
                                                                         : "no")
                << " uuid=" << uuid_hex(props[i].uuid)
                << " maxAllocMiB=" << (props[i].maxMemAllocSize >> 20)
                << " extExport=" << external_flags(ext.memoryAllocationExportTypes)
                << " extImport=" << external_flags(ext.memoryAllocationImportTypes)
                << "\n";
    }

    std::cout << "p2p_matrix source->peer can_access flags\n";
    for (uint32_t i = 0; i < device_count; ++i) {
      for (uint32_t j = 0; j < device_count; ++j) {
        ze_bool_t can_access = false;
        ze_result_t access_result =
            zeDeviceCanAccessPeer(devices[i], devices[j], &can_access);

        ze_device_p2p_properties_t p2p = {};
        p2p.stype = ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES;
        ze_result_t p2p_result = zeDeviceGetP2PProperties(devices[i], devices[j], &p2p);

        std::cout << "p2p[" << i << "->" << j << "]"
                  << " can_access_result=" << result_name(access_result)
                  << " can_access=" << (can_access ? "yes" : "no")
                  << " props_result=" << result_name(p2p_result)
                  << " flags=" << p2p_flags(p2p.flags) << "\n";
      }
    }

    if (do_p2p_fill_test) {
      run_p2p_fill_test(drivers[d], devices);
    }
    if (do_ipc_fork_test) {
      run_ipc_fork_test(argv[0], drivers[d], devices);
    }
  }

  return 0;
}
