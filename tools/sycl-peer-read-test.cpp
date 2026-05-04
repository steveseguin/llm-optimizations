#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

int main() {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    const size_t ndev = std::min<size_t>(devices.size(), 4);
    if (ndev < 2) {
        std::fprintf(stderr, "need at least 2 gpu devices, found %zu\n", devices.size());
        return 2;
    }

    std::vector<std::unique_ptr<sycl::queue>> queues;
    queues.reserve(ndev);
    for (size_t d = 0; d < ndev; ++d) {
        queues.emplace_back(new sycl::queue(devices[d], sycl::property::queue::in_order{}));
    }
    constexpr size_t n = 5120;
    constexpr int repeats = 200;

    std::vector<float *> ptrs(ndev, nullptr);
    for (size_t d = 0; d < ndev; ++d) {
        ptrs[d] = sycl::malloc_device<float>(n, *queues[d]);
        if (ptrs[d] == nullptr) {
            std::fprintf(stderr, "device allocation failed on %zu\n", d);
            return 3;
        }
    }

    std::vector<std::vector<float>> host(ndev, std::vector<float>(n));

    size_t bad = 0;
    for (int r = 0; r < repeats; ++r) {
        std::vector<sycl::event> ready(ndev);
        for (size_t d = 0; d < ndev; ++d) {
            float * dst = ptrs[d];
            ready[d] = queues[d]->submit([&](sycl::handler & h) {
                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                    const size_t i = idx[0];
                    dst[i] = (float) (int(i % (251 - 17*d)) + r*int(d + 1));
                });
            });
        }

        float * p0 = ptrs[0];
        float * p1 = ptrs[1];
        float * p2 = ndev > 2 ? ptrs[2] : nullptr;
        float * p3 = ndev > 3 ? ptrs[3] : nullptr;

        sycl::event reduce = queues[0]->submit([&](sycl::handler & h) {
            h.depends_on(ready);
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                const size_t i = idx[0];
                float sum = p0[i] + p1[i];
                if (ndev > 2) {
                    sum += p2[i];
                }
                if (ndev > 3) {
                    sum += p3[i];
                }
                p0[i] = sum;
                p1[i] = sum;
                if (ndev > 2) {
                    p2[i] = sum;
                }
                if (ndev > 3) {
                    p3[i] = sum;
                }
            });
        });

        std::vector<sycl::event> seen(ndev);
        seen[0] = reduce;
        for (size_t d = 1; d < ndev; ++d) {
            seen[d] = queues[d]->submit([&](sycl::handler & h) {
                h.depends_on(reduce);
                h.single_task([=]() {});
            });
        }

        for (size_t d = 0; d < ndev; ++d) {
            float * src = ptrs[d];
            float * dst = host[d].data();
            queues[d]->submit([&](sycl::handler & h) {
                h.depends_on(seen[d]);
                h.memcpy(dst, src, n*sizeof(float));
            }).wait();
        }

        for (size_t i = 0; i < n; ++i) {
            float expected = 0.0f;
            for (size_t d = 0; d < ndev; ++d) {
                expected += (float) (int(i % (251 - 17*d)) + r*int(d + 1));
            }
            for (size_t d = 0; d < ndev; ++d) {
                if (host[d][i] != expected) {
                    if (bad < 10) {
                        std::fprintf(stderr,
                                     "bad r=%d d=%zu i=%zu value=%f expected=%f\n",
                                     r, d, i, host[d][i], expected);
                    }
                    bad++;
                }
            }
        }
    }

    for (size_t d = 0; d < ndev; ++d) {
        sycl::free(ptrs[d], *queues[d]);
    }

    if (bad != 0) {
        std::fprintf(stderr, "bad values: %zu\n", bad);
        return 4;
    }
    std::printf("peer kernel read ok across %zu devices\n", ndev);
    return 0;
}
