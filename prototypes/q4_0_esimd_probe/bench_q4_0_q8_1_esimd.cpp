#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;

static constexpr int QK4_0 = 32;
static constexpr int QK8_1 = 32;

#ifndef Q4_ESIMD_BIAS_IN_ACC
#define Q4_ESIMD_BIAS_IN_ACC 0
#endif

#ifndef Q4_ESIMD_BLOCK_LOAD_SCALES
#define Q4_ESIMD_BLOCK_LOAD_SCALES 0
#endif

struct options {
    int n = 17408;
    int k = 5120;
    int iters = 300;
    int warmup = 40;
    int check_rows = 8;
    bool fused2 = false;
};

static int parse_int_arg(const char * arg, const char * name) {
    char * end = nullptr;
    long v = std::strtol(arg, &end, 10);
    if (!end || *end != '\0' || v <= 0 || v > INT32_MAX) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + arg);
    }
    return static_cast<int>(v);
}

static options parse_args(int argc, char ** argv) {
    options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (a == "--n") {
            opt.n = parse_int_arg(need_value("--n"), "--n");
        } else if (a == "--k") {
            opt.k = parse_int_arg(need_value("--k"), "--k");
        } else if (a == "--iters") {
            opt.iters = parse_int_arg(need_value("--iters"), "--iters");
        } else if (a == "--warmup") {
            opt.warmup = parse_int_arg(need_value("--warmup"), "--warmup");
        } else if (a == "--check-rows") {
            opt.check_rows = parse_int_arg(need_value("--check-rows"), "--check-rows");
        } else if (a == "--fused2") {
            opt.fused2 = true;
        } else if (a == "--help" || a == "-h") {
            std::cout << "usage: bench_q4_0_q8_1_esimd [--n rows] [--k cols] [--iters n] [--warmup n] [--check-rows n] [--fused2]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown arg: " + a);
        }
    }
    if (opt.k % 128 != 0) {
        throw std::runtime_error("--k must be a multiple of 128");
    }
    return opt;
}

static void quantize_q8_1(
        const std::vector<float> & x,
        std::vector<int8_t> & q8,
        std::vector<sycl::half> & ds) {
    const int k = (int) x.size();
    const int nblocks = k / QK8_1;
    q8.resize(k);
    ds.resize((size_t) nblocks * 2);

    for (int b = 0; b < nblocks; ++b) {
        float amax = 0.0f;
        float sum = 0.0f;
        for (int j = 0; j < QK8_1; ++j) {
            const float v = x[b * QK8_1 + j];
            amax = std::max(amax, std::fabs(v));
            sum += v;
        }
        const float d = amax == 0.0f ? 0.0f : amax / 127.0f;
        for (int j = 0; j < QK8_1; ++j) {
            int q = d == 0.0f ? 0 : (int) std::nearbyint(x[b * QK8_1 + j] / d);
            q = std::max(-127, std::min(127, q));
            q8[b * QK8_1 + j] = (int8_t) q;
        }
        ds[(size_t) b * 2 + 0] = sycl::half(d);
        ds[(size_t) b * 2 + 1] = sycl::half(sum);
    }
}

static void fill_q4_0(
        std::vector<uint8_t> & qs,
        std::vector<sycl::half> & d,
        int n,
        int k,
        uint32_t seed) {
    const int nblocks = k / QK4_0;
    qs.resize((size_t) n * k / 2);
    d.resize((size_t) n * nblocks);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> qdist(0, 15);
    std::uniform_real_distribution<float> ddist(-0.18f, 0.18f);

    for (int row = 0; row < n; ++row) {
        for (int b = 0; b < nblocks; ++b) {
            float scale = ddist(rng);
            if (std::fabs(scale) < 0.01f) {
                scale = std::copysign(0.01f, scale == 0.0f ? 1.0f : scale);
            }
            d[(size_t) row * nblocks + b] = sycl::half(scale);
            uint8_t * block = qs.data() + (size_t) row * (k / 2) + b * (QK4_0 / 2);
            for (int j = 0; j < QK4_0 / 2; ++j) {
                const uint8_t lo = (uint8_t) qdist(rng);
                const uint8_t hi = (uint8_t) qdist(rng);
                block[j] = (uint8_t) (lo | (hi << 4));
            }
        }
    }
}

static void q4_0_q8_1_ref_row(
        const int8_t * q8,
        const sycl::half * q8_ds,
        const uint8_t * qs,
        const sycl::half * d,
        float * out,
        int n,
        int k) {
    const int nblocks = k / QK4_0;
    for (int row = 0; row < n; ++row) {
        float acc = 0.0f;
        const uint8_t * row_qs = qs + (size_t) row * (k / 2);
        const sycl::half * row_d = d + (size_t) row * nblocks;
        for (int b = 0; b < nblocks; ++b) {
            const float d4 = (float) row_d[b];
            const float d8 = (float) q8_ds[(size_t) b * 2 + 0];
            const float s8 = (float) q8_ds[(size_t) b * 2 + 1];
            int sumi = 0;
            const uint8_t * bq = row_qs + b * (QK4_0 / 2);
            const int8_t * bx = q8 + b * QK8_1;
            for (int j = 0; j < QK4_0 / 2; ++j) {
                const uint8_t packed = bq[j];
                sumi += (packed & 0x0f) * (int) bx[j];
                sumi += ((packed >> 4) & 0x0f) * (int) bx[j + 16];
            }
            acc += d4 * (sumi * d8 - 8.0f * s8);
        }
        out[row] = acc;
    }
}

struct q4_0_q8_1_esimd_kernel {
    const int8_t * q8;
    const sycl::half * q8_ds;
    const uint8_t * qs;
    const sycl::half * d;
    float * out;
    int n;
    int k;
    int nblocks;

    void operator()(sycl::id<1> tid) const SYCL_ESIMD_KERNEL {
        const int row = (int) tid[0];
        if (row >= n) {
            return;
        }

        esimd::simd<float, 64> acc(0.0f);
        float bias = 0.0f;
        const uint8_t * row_qs = qs + (size_t) row * (k / 2);
        const sycl::half * row_d = d + (size_t) row * nblocks;

        for (int kk = 0; kk < k; kk += 128) {
            esimd::simd<int8_t, 128> xv = esimd::block_load<int8_t, 128>(q8 + kk);
            esimd::simd<uint8_t, 64> raw = esimd::block_load<uint8_t, 64>(row_qs + kk / 2);
#if Q4_ESIMD_BLOCK_LOAD_SCALES
            const int block0 = kk / QK4_0;
            const esimd::simd<float, 4> d4v =
                esimd::convert<float>(esimd::block_load<sycl::half, 4>(row_d + block0));
            const esimd::simd<float, 8> q8v =
                esimd::convert<float>(esimd::block_load<sycl::half, 8>(q8_ds + (size_t) block0 * 2));
#endif

#pragma unroll
            for (int blk = 0; blk < 4; ++blk) {
                const int qoff = blk * 16;
                const int xoff = blk * 32;
                const int block = kk / QK4_0 + blk;
#if Q4_ESIMD_BLOCK_LOAD_SCALES
                const float d4 = (float) d4v[blk];
                const float d8 = (float) q8v[blk * 2 + 0];
                const float s8 = (float) q8v[blk * 2 + 1];
#else
                const float d4 = (float) row_d[block];
                const float d8 = (float) q8_ds[(size_t) block * 2 + 0];
                const float s8 = (float) q8_ds[(size_t) block * 2 + 1];
#endif
                const float scale = d4 * d8;

                const esimd::simd<uint16_t, 16> packed =
                    esimd::convert<uint16_t>(raw.template select<16, 1>(qoff).read());
                const esimd::simd<float, 16> w_lo = esimd::convert<float>(packed & 0x000f);
                const esimd::simd<float, 16> w_hi = esimd::convert<float>((packed >> 4) & 0x000f);
                const esimd::simd<float, 16> x_lo =
                    esimd::convert<float>(xv.template select<16, 1>(xoff).read());
                const esimd::simd<float, 16> x_hi =
                    esimd::convert<float>(xv.template select<16, 1>(xoff + 16).read());

                acc.template select<16, 1>(blk * 16) += (w_lo * x_lo + w_hi * x_hi) * scale;
#if Q4_ESIMD_BIAS_IN_ACC
                acc[blk * 16] += -8.0f * d4 * s8;
#else
                bias += -8.0f * d4 * s8;
#endif
            }
        }

        acc.template select<32, 1>(0) += acc.template select<32, 1>(32);
        acc.template select<16, 1>(0) += acc.template select<16, 1>(16);
        acc.template select<8, 1>(0) += acc.template select<8, 1>(8);
        acc.template select<4, 1>(0) += acc.template select<4, 1>(4);
        acc.template select<2, 1>(0) += acc.template select<2, 1>(2);
        out[row] = (float) acc[0] + (float) acc[1] + bias;
    }
};

struct q4_0_q8_1_esimd_fused2_kernel {
    const int8_t * q8;
    const sycl::half * q8_ds;
    const uint8_t * qs0;
    const sycl::half * d0;
    float * out0;
    const uint8_t * qs1;
    const sycl::half * d1;
    float * out1;
    int n;
    int k;
    int nblocks;

    void operator()(sycl::id<1> tid) const SYCL_ESIMD_KERNEL {
        const int row = (int) tid[0];
        if (row >= n) {
            return;
        }

        esimd::simd<float, 64> acc0(0.0f);
        esimd::simd<float, 64> acc1(0.0f);
        float bias0 = 0.0f;
        float bias1 = 0.0f;
        const uint8_t * row_qs0 = qs0 + (size_t) row * (k / 2);
        const uint8_t * row_qs1 = qs1 + (size_t) row * (k / 2);
        const sycl::half * row_d0 = d0 + (size_t) row * nblocks;
        const sycl::half * row_d1 = d1 + (size_t) row * nblocks;

        for (int kk = 0; kk < k; kk += 128) {
            esimd::simd<int8_t, 128> xv = esimd::block_load<int8_t, 128>(q8 + kk);
            esimd::simd<uint8_t, 64> raw0 = esimd::block_load<uint8_t, 64>(row_qs0 + kk / 2);
            esimd::simd<uint8_t, 64> raw1 = esimd::block_load<uint8_t, 64>(row_qs1 + kk / 2);
#if Q4_ESIMD_BLOCK_LOAD_SCALES
            const int block0 = kk / QK4_0;
            const esimd::simd<float, 4> d4v0 =
                esimd::convert<float>(esimd::block_load<sycl::half, 4>(row_d0 + block0));
            const esimd::simd<float, 4> d4v1 =
                esimd::convert<float>(esimd::block_load<sycl::half, 4>(row_d1 + block0));
            const esimd::simd<float, 8> q8v =
                esimd::convert<float>(esimd::block_load<sycl::half, 8>(q8_ds + (size_t) block0 * 2));
#endif

#pragma unroll
            for (int blk = 0; blk < 4; ++blk) {
                const int qoff = blk * 16;
                const int xoff = blk * 32;
                const int block = kk / QK4_0 + blk;
#if Q4_ESIMD_BLOCK_LOAD_SCALES
                const float d8 = (float) q8v[blk * 2 + 0];
                const float s8 = (float) q8v[blk * 2 + 1];
#else
                const float d8 = (float) q8_ds[(size_t) block * 2 + 0];
                const float s8 = (float) q8_ds[(size_t) block * 2 + 1];
#endif
                const esimd::simd<float, 16> x_lo =
                    esimd::convert<float>(xv.template select<16, 1>(xoff).read());
                const esimd::simd<float, 16> x_hi =
                    esimd::convert<float>(xv.template select<16, 1>(xoff + 16).read());

#if Q4_ESIMD_BLOCK_LOAD_SCALES
                const float d4_0 = (float) d4v0[blk];
#else
                const float d4_0 = (float) row_d0[block];
#endif
                const esimd::simd<uint16_t, 16> packed0 =
                    esimd::convert<uint16_t>(raw0.template select<16, 1>(qoff).read());
                const esimd::simd<float, 16> w0_lo = esimd::convert<float>(packed0 & 0x000f);
                const esimd::simd<float, 16> w0_hi = esimd::convert<float>((packed0 >> 4) & 0x000f);
                acc0.template select<16, 1>(blk * 16) += (w0_lo * x_lo + w0_hi * x_hi) * (d4_0 * d8);
#if Q4_ESIMD_BIAS_IN_ACC
                acc0[blk * 16] += -8.0f * d4_0 * s8;
#else
                bias0 += -8.0f * d4_0 * s8;
#endif

#if Q4_ESIMD_BLOCK_LOAD_SCALES
                const float d4_1 = (float) d4v1[blk];
#else
                const float d4_1 = (float) row_d1[block];
#endif
                const esimd::simd<uint16_t, 16> packed1 =
                    esimd::convert<uint16_t>(raw1.template select<16, 1>(qoff).read());
                const esimd::simd<float, 16> w1_lo = esimd::convert<float>(packed1 & 0x000f);
                const esimd::simd<float, 16> w1_hi = esimd::convert<float>((packed1 >> 4) & 0x000f);
                acc1.template select<16, 1>(blk * 16) += (w1_lo * x_lo + w1_hi * x_hi) * (d4_1 * d8);
#if Q4_ESIMD_BIAS_IN_ACC
                acc1[blk * 16] += -8.0f * d4_1 * s8;
#else
                bias1 += -8.0f * d4_1 * s8;
#endif
            }
        }

        acc0.template select<32, 1>(0) += acc0.template select<32, 1>(32);
        acc0.template select<16, 1>(0) += acc0.template select<16, 1>(16);
        acc0.template select<8, 1>(0) += acc0.template select<8, 1>(8);
        acc0.template select<4, 1>(0) += acc0.template select<4, 1>(4);
        acc0.template select<2, 1>(0) += acc0.template select<2, 1>(2);
        out0[row] = (float) acc0[0] + (float) acc0[1] + bias0;

        acc1.template select<32, 1>(0) += acc1.template select<32, 1>(32);
        acc1.template select<16, 1>(0) += acc1.template select<16, 1>(16);
        acc1.template select<8, 1>(0) += acc1.template select<8, 1>(8);
        acc1.template select<4, 1>(0) += acc1.template select<4, 1>(4);
        acc1.template select<2, 1>(0) += acc1.template select<2, 1>(2);
        out1[row] = (float) acc1[0] + (float) acc1[1] + bias1;
    }
};

static sycl::event launch_single(
        sycl::queue & q,
        const int8_t * q8,
        const sycl::half * q8_ds,
        const uint8_t * qs,
        const sycl::half * d,
        float * out,
        int n,
        int k) {
    const int nblocks = k / QK4_0;
    return q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(n), q4_0_q8_1_esimd_kernel{q8, q8_ds, qs, d, out, n, k, nblocks});
    });
}

static sycl::event launch_fused2(
        sycl::queue & q,
        const int8_t * q8,
        const sycl::half * q8_ds,
        const uint8_t * qs0,
        const sycl::half * d0,
        float * out0,
        const uint8_t * qs1,
        const sycl::half * d1,
        float * out1,
        int n,
        int k) {
    const int nblocks = k / QK4_0;
    return q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(n),
                       q4_0_q8_1_esimd_fused2_kernel{q8, q8_ds, qs0, d0, out0, qs1, d1, out1, n, k, nblocks});
    });
}

static double event_us(const sycl::event & ev) {
    const auto start = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto end = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
    return (double) (end - start) / 1000.0;
}

template <typename F>
static double median_time_us(int warmup, int iters, F launch) {
    for (int i = 0; i < warmup; ++i) {
        launch().wait();
    }
    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        sycl::event ev = launch();
        ev.wait();
        samples.push_back(event_us(ev));
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

int main(int argc, char ** argv) {
    try {
        const options opt = parse_args(argc, argv);
        const int nblocks = opt.k / QK4_0;
        const size_t qs_bytes = (size_t) opt.n * opt.k / 2;
        const size_t d_count = (size_t) opt.n * nblocks;

        std::vector<float> x(opt.k);
        std::mt19937 rng(1);
        std::normal_distribution<float> xdist(0.0f, 1.0f);
        for (float & v : x) {
            v = xdist(rng);
        }

        std::vector<int8_t> h_q8;
        std::vector<sycl::half> h_q8_ds;
        quantize_q8_1(x, h_q8, h_q8_ds);

        std::vector<uint8_t> h_qs0;
        std::vector<sycl::half> h_d0;
        std::vector<uint8_t> h_qs1;
        std::vector<sycl::half> h_d1;
        fill_q4_0(h_qs0, h_d0, opt.n, opt.k, 2);
        fill_q4_0(h_qs1, h_d1, opt.n, opt.k, 3);

        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});
        std::cout << "device=" << q.get_device().get_info<sycl::info::device::name>() << "\n";

        int8_t * d_q8 = sycl::malloc_device<int8_t>(h_q8.size(), q);
        sycl::half * d_q8_ds = sycl::malloc_device<sycl::half>(h_q8_ds.size(), q);
        uint8_t * d_qs0 = sycl::malloc_device<uint8_t>(h_qs0.size(), q);
        sycl::half * d_d0 = sycl::malloc_device<sycl::half>(h_d0.size(), q);
        uint8_t * d_qs1 = sycl::malloc_device<uint8_t>(h_qs1.size(), q);
        sycl::half * d_d1 = sycl::malloc_device<sycl::half>(h_d1.size(), q);
        float * d_out0 = sycl::malloc_device<float>(opt.n, q);
        float * d_out1 = sycl::malloc_device<float>(opt.n, q);

        q.memcpy(d_q8, h_q8.data(), h_q8.size() * sizeof(int8_t)).wait();
        q.memcpy(d_q8_ds, h_q8_ds.data(), h_q8_ds.size() * sizeof(sycl::half)).wait();
        q.memcpy(d_qs0, h_qs0.data(), h_qs0.size() * sizeof(uint8_t)).wait();
        q.memcpy(d_d0, h_d0.data(), h_d0.size() * sizeof(sycl::half)).wait();
        q.memcpy(d_qs1, h_qs1.data(), h_qs1.size() * sizeof(uint8_t)).wait();
        q.memcpy(d_d1, h_d1.data(), h_d1.size() * sizeof(sycl::half)).wait();

        if (opt.fused2) {
            launch_fused2(q, d_q8, d_q8_ds, d_qs0, d_d0, d_out0, d_qs1, d_d1, d_out1, opt.n, opt.k).wait();
        } else {
            launch_single(q, d_q8, d_q8_ds, d_qs0, d_d0, d_out0, opt.n, opt.k).wait();
        }

        std::vector<float> out0(opt.n);
        std::vector<float> out1(opt.n);
        q.memcpy(out0.data(), d_out0, opt.n * sizeof(float)).wait();
        if (opt.fused2) {
            q.memcpy(out1.data(), d_out1, opt.n * sizeof(float)).wait();
        }

        const int rows_to_check = std::min(opt.check_rows, opt.n);
        std::vector<float> ref0(rows_to_check);
        std::vector<float> ref1(rows_to_check);
        q4_0_q8_1_ref_row(h_q8.data(), h_q8_ds.data(), h_qs0.data(), h_d0.data(),
                           ref0.data(), rows_to_check, opt.k);
        if (opt.fused2) {
            q4_0_q8_1_ref_row(h_q8.data(), h_q8_ds.data(), h_qs1.data(), h_d1.data(),
                               ref1.data(), rows_to_check, opt.k);
        }

        double max_abs = 0.0;
        double max_rel = 0.0;
        auto check = [&](const std::vector<float> & got, const std::vector<float> & ref) {
            for (int i = 0; i < rows_to_check; ++i) {
                const double abs_err = std::fabs((double) got[i] - ref[i]);
                const double rel_err = abs_err / std::max(1.0, std::fabs((double) ref[i]));
                max_abs = std::max(max_abs, abs_err);
                max_rel = std::max(max_rel, rel_err);
            }
        };
        check(out0, ref0);
        if (opt.fused2) {
            check(out1, ref1);
        }

        const double us = opt.fused2
            ? median_time_us(opt.warmup, opt.iters, [&]() {
                return launch_fused2(q, d_q8, d_q8_ds, d_qs0, d_d0, d_out0, d_qs1, d_d1, d_out1, opt.n, opt.k);
            })
            : median_time_us(opt.warmup, opt.iters, [&]() {
                return launch_single(q, d_q8, d_q8_ds, d_qs0, d_d0, d_out0, opt.n, opt.k);
            });

        const double matvecs = opt.fused2 ? 2.0 : 1.0;
        const double ops = matvecs * 2.0 * (double) opt.n * (double) opt.k;
        const double q4_bytes = matvecs * (double) (qs_bytes + d_count * sizeof(sycl::half));
        const double q8_bytes = (double) (h_q8.size() * sizeof(int8_t) + h_q8_ds.size() * sizeof(sycl::half));
        const double gbps = (q4_bytes + q8_bytes) / (us * 1.0e-6) / 1.0e9;
        const double gops = ops / (us * 1.0e-6) / 1.0e9;

        std::cout << std::fixed << std::setprecision(3)
                  << "mode=" << (opt.fused2 ? "fused2" : "single")
                  << " n=" << opt.n
                  << " k=" << opt.k
                  << " median_us=" << us
                  << " gops=" << gops
                  << " bytes_gbps=" << gbps
                  << " max_abs=" << max_abs
                  << " max_rel=" << max_rel
                  << "\n";

        sycl::free(d_q8, q);
        sycl::free(d_q8_ds, q);
        sycl::free(d_qs0, q);
        sycl::free(d_d0, q);
        sycl::free(d_qs1, q);
        sycl::free(d_d1, q);
        sycl::free(d_out0, q);
        sycl::free(d_out1, q);
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
