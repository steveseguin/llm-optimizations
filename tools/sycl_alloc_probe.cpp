#include <sycl/sycl.hpp>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const bool all_devices = argc > 1 && std::string(argv[1]) == "all";
    const int device_index = !all_devices && argc > 1 ? std::atoi(argv[1]) : 0;
    const int size_arg = all_devices ? 2 : 2;
    const std::vector<double> gib_tests = argc > size_arg
        ? std::vector<double>{std::atof(argv[2])}
        : std::vector<double>{8.0, 12.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0};

    auto platforms = sycl::platform::get_platforms();
    std::vector<sycl::device> devices;
    for (const auto & platform : platforms) {
        for (const auto & device : platform.get_devices()) {
            if (device.is_gpu()) {
                devices.push_back(device);
            }
        }
    }

    if (!all_devices && (device_index < 0 || device_index >= static_cast<int>(devices.size()))) {
        std::cerr << "invalid device index " << device_index << ", gpu_count=" << devices.size() << "\n";
        return 2;
    }

    if (all_devices) {
        sycl::context context(devices);
        for (double gib : gib_tests) {
            const size_t bytes = static_cast<size_t>(gib * 1024.0 * 1024.0 * 1024.0);
            std::vector<sycl::queue> queues;
            std::vector<void *> ptrs;
            std::cout << "alloc_all_test=" << gib << " GiB per GPU across " << devices.size() << " GPUs ... " << std::flush;
            try {
                for (size_t idev = 0; idev < devices.size(); ++idev) {
                    const sycl::device & device = devices[idev];
                    queues.emplace_back(context, device);
                    void * ptr = sycl::malloc_device(bytes, device, context);
                    if (ptr == nullptr) {
                        throw std::runtime_error("malloc_device returned null at gpu " + std::to_string(idev));
                    }
                    ptrs.push_back(ptr);
                }
                for (size_t i = 0; i < ptrs.size(); ++i) {
                    queues[i].memset(ptrs[i], 0, 4096).wait();
                }
                std::cout << "ok\n";
            } catch (const std::exception & e) {
                std::cout << "failed: " << e.what() << "\n";
            }
            for (size_t i = 0; i < ptrs.size(); ++i) {
                sycl::free(ptrs[i], queues[i]);
            }
            for (sycl::queue & queue : queues) {
                queue.wait();
            }
        }
        return 0;
    }

    sycl::device device = devices[device_index];
    sycl::queue queue(device);

    std::cout << "device=" << device.get_info<sycl::info::device::name>() << "\n";
    std::cout << "global_mem="
              << device.get_info<sycl::info::device::global_mem_size>() / 1024.0 / 1024.0 / 1024.0
              << " GiB\n";
    std::cout << "max_mem_alloc="
              << device.get_info<sycl::info::device::max_mem_alloc_size>() / 1024.0 / 1024.0 / 1024.0
              << " GiB\n";

    for (double gib : gib_tests) {
        const size_t bytes = static_cast<size_t>(gib * 1024.0 * 1024.0 * 1024.0);
        std::cout << "alloc_test=" << gib << " GiB ... " << std::flush;
        try {
            void * ptr = sycl::malloc_device(bytes, queue);
            if (ptr == nullptr) {
                std::cout << "null\n";
                continue;
            }
            queue.memset(ptr, 0, 4096).wait();
            sycl::free(ptr, queue);
            queue.wait();
            std::cout << "ok\n";
        } catch (const sycl::exception & e) {
            std::cout << "failed: " << e.what() << "\n";
        }
    }

    return 0;
}
