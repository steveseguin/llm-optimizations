#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;

static constexpr int QK4_0 = 32;
static constexpr int TILE_K = 128;

struct args_t {
    int n = 17408;
    int k = 5120;
    int ks = 1;
    int iters = 200;
    int warmup = 20;
    int seed = 1;
};

static int parse_int_arg(int argc, char ** argv, const char * name, int def) {
    const std::string key = std::string("--") + name + "=";
    for (int i = 1; i < argc; ++i) {
        const std::string cur(argv[i]);
        if (cur.rfind(key, 0) == 0) {
            return std::stoi(cur.substr(key.size()));
        }
    }
    return def;
}

static args_t parse_args(int argc, char ** argv) {
    args_t args;
    args.n = parse_int_arg(argc, argv, "n", args.n);
    args.k = parse_int_arg(argc, argv, "k", args.k);
    args.ks = parse_int_arg(argc, argv, "ks", args.ks);
    args.iters = parse_int_arg(argc, argv, "iters", args.iters);
    args.warmup = parse_int_arg(argc, argv, "warmup", args.warmup);
    args.seed = parse_int_arg(argc, argv, "seed", args.seed);
    if (args.n <= 0 || args.k <= 0 || args.iters <= 0 || args.warmup < 0) {
        throw std::runtime_error("invalid non-positive argument");
    }
    if (args.k % TILE_K != 0) {
        throw std::runtime_error("k must be a multiple of 128 for this first ESIMD prototype");
    }
    if (!(args.ks == 1 || args.ks == 2 || args.ks == 4 || args.ks == 8)) {
        throw std::runtime_error("ks must be one of 1, 2, 4, 8");
    }
    if (args.k % args.ks != 0 || (args.k / args.ks) % TILE_K != 0) {
        throw std::runtime_error("k / ks must be a multiple of 128");
    }
    return args;
}

static void quantize_q8_1_host(
        const std::vector<float> & x,
        std::vector<int8_t> & q8_qs,
        std::vector<sycl::half> & q8_d,
        std::vector<sycl::half> & q8_s) {
    const int k = static_cast<int>(x.size());
    const int nb = k / QK4_0;
    q8_qs.resize(k);
    q8_d.resize(nb);
    q8_s.resize(nb);

    for (int b = 0; b < nb; ++b) {
        float amax = 0.0f;
        float sum = 0.0f;
        for (int i = 0; i < QK4_0; ++i) {
            const float v = x[b * QK4_0 + i];
            amax = std::max(amax, std::fabs(v));
            sum += v;
        }
        const float d = amax == 0.0f ? 0.0f : amax / 127.0f;
        q8_d[b] = sycl::half(d);
        q8_s[b] = sycl::half(sum);
        for (int i = 0; i < QK4_0; ++i) {
            int q = d == 0.0f ? 0 : static_cast<int>(std::nearbyint(x[b * QK4_0 + i] / d));
            q = std::max(-127, std::min(127, q));
            q8_qs[b * QK4_0 + i] = static_cast<int8_t>(q);
        }
    }
}

static void cpu_ref_q4_0_q8_1(
        const std::vector<uint8_t> & q4_qs,
        const std::vector<sycl::half> & q4_d,
        const std::vector<int8_t> & q8_qs,
        const std::vector<sycl::half> & q8_d,
        const std::vector<sycl::half> & q8_s,
        int n,
        int k,
        std::vector<float> & out) {
    const int nb = k / QK4_0;
    out.assign(n, 0.0f);
    for (int row = 0; row < n; ++row) {
        float acc = 0.0f;
        const uint8_t * row_qs = q4_qs.data() + static_cast<size_t>(row) * (k / 2);
        const sycl::half * row_d = q4_d.data() + static_cast<size_t>(row) * nb;
        for (int b = 0; b < nb; ++b) {
            int dot = 0;
            const int base = b * QK4_0;
            const uint8_t * packed = row_qs + b * (QK4_0 / 2);
            for (int j = 0; j < QK4_0 / 2; ++j) {
                const uint8_t byte = packed[j];
                const int lo = byte & 0x0f;
                const int hi = (byte >> 4) & 0x0f;
                dot += lo * static_cast<int>(q8_qs[base + 2 * j + 0]);
                dot += hi * static_cast<int>(q8_qs[base + 2 * j + 1]);
            }
            const float d4 = static_cast<float>(row_d[b]);
            const float d8 = static_cast<float>(q8_d[b]);
            const float s8 = static_cast<float>(q8_s[b]);
            acc += d4 * (static_cast<float>(dot) * d8 - 8.0f * s8);
        }
        out[row] = acc;
    }
}

template <int VL, int K_SPLIT>
struct q4_0_q8_1_gemv_kernel {
    const uint8_t * q4_qs;
    const sycl::half * q4_d;
    const int8_t * q8_qs;
    const sycl::half * q8_d;
    const sycl::half * q8_s;
    float * out;
    int n;
    int k;
    int nb;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            esimd::slm_init<K_SPLIT * sizeof(float)>();
        }

        const int row = item.get_group(0);
        const int lid = item.get_local_id(0);
        if (row >= n) {
            return;
        }

        const uint8_t * row_qs = q4_qs + static_cast<size_t>(row) * (k / 2);
        const sycl::half * row_d = q4_d + static_cast<size_t>(row) * nb;
        float acc = 0.0f;

        const int kp = k / K_SPLIT;
        const int k_begin = lid * kp;
        const int k_end = k_begin + kp;

        for (int kk = k_begin; kk < k_end; kk += VL) {
            esimd::simd<uint8_t, VL / 2> raw = esimd::block_load<uint8_t, VL / 2>(row_qs + kk / 2);
            esimd::simd<int8_t, VL> q8 = esimd::block_load<int8_t, VL>(q8_qs + kk);

#pragma unroll
            for (int sub = 0; sub < VL / QK4_0; ++sub) {
                constexpr int Q4_BYTES = QK4_0 / 2;
                esimd::simd<uint8_t, Q4_BYTES> packed = raw.template select<Q4_BYTES, 1>(sub * Q4_BYTES);
                esimd::simd<uint16_t, Q4_BYTES> packed16 = esimd::convert<uint16_t>(packed);
                esimd::simd<float, Q4_BYTES> lo = esimd::convert<float>(packed16 & 0x000f);
                esimd::simd<float, Q4_BYTES> hi = esimd::convert<float>((packed16 >> 4) & 0x000f);

                esimd::simd<int8_t, Q4_BYTES> q8_even_i = q8.template select<Q4_BYTES, 2>(sub * QK4_0);
                esimd::simd<int8_t, Q4_BYTES> q8_odd_i = q8.template select<Q4_BYTES, 2>(sub * QK4_0 + 1);
                esimd::simd<float, Q4_BYTES> q8_even = esimd::convert<float>(q8_even_i);
                esimd::simd<float, Q4_BYTES> q8_odd = esimd::convert<float>(q8_odd_i);

                const int block = kk / QK4_0 + sub;
                const float dot = esimd::reduce<float>(lo * q8_even + hi * q8_odd, std::plus<>());
                const float d4 = static_cast<float>(row_d[block]);
                const float d8 = static_cast<float>(q8_d[block]);
                const float s8 = static_cast<float>(q8_s[block]);
                acc += d4 * (dot * d8 - 8.0f * s8);
            }
        }

        if constexpr (K_SPLIT == 1) {
            out[row] = acc;
        } else {
            esimd::slm_block_store<float, 1>(lid * sizeof(float), esimd::simd<float, 1>(acc));
            esimd::barrier();
            if (lid == 0) {
                esimd::simd<float, K_SPLIT> parts = esimd::slm_block_load<float, K_SPLIT>(0);
                out[row] = esimd::reduce<float>(parts, std::plus<>());
            }
        }
    }
};

template <int K_SPLIT>
static void run_esimd(
        sycl::queue & q,
        const uint8_t * d_q4_qs,
        const sycl::half * d_q4_d,
        const int8_t * d_q8_qs,
        const sycl::half * d_q8_d,
        const sycl::half * d_q8_s,
        float * d_out,
        int n,
        int k) {
    const int nb = k / QK4_0;
    sycl::nd_range<1> grid(sycl::range<1>(n * K_SPLIT), sycl::range<1>(K_SPLIT));
    q.submit([&](sycl::handler & cgh) {
        q4_0_q8_1_gemv_kernel<TILE_K, K_SPLIT> kernel{d_q4_qs, d_q4_d, d_q8_qs, d_q8_d, d_q8_s, d_out, n, k, nb};
        cgh.parallel_for(grid, kernel);
    });
}

static void run_esimd_ks(
        sycl::queue & q,
        const uint8_t * d_q4_qs,
        const sycl::half * d_q4_d,
        const int8_t * d_q8_qs,
        const sycl::half * d_q8_d,
        const sycl::half * d_q8_s,
        float * d_out,
        int n,
        int k,
        int ks) {
    switch (ks) {
        case 1:
            run_esimd<1>(q, d_q4_qs, d_q4_d, d_q8_qs, d_q8_d, d_q8_s, d_out, n, k);
            break;
        case 2:
            run_esimd<2>(q, d_q4_qs, d_q4_d, d_q8_qs, d_q8_d, d_q8_s, d_out, n, k);
            break;
        case 4:
            run_esimd<4>(q, d_q4_qs, d_q4_d, d_q8_qs, d_q8_d, d_q8_s, d_out, n, k);
            break;
        case 8:
            run_esimd<8>(q, d_q4_qs, d_q4_d, d_q8_qs, d_q8_d, d_q8_s, d_out, n, k);
            break;
        default:
            throw std::runtime_error("unsupported ks");
    }
}

int main(int argc, char ** argv) {
    try {
        const args_t args = parse_args(argc, argv);
        const int n = args.n;
        const int k = args.k;
        const int nb = k / QK4_0;

        std::mt19937 rng(args.seed);
        std::uniform_real_distribution<float> x_dist(-2.0f, 2.0f);
        std::uniform_int_distribution<int> q4_dist(0, 255);
        std::uniform_real_distribution<float> d_dist(-0.05f, -0.001f);

        std::vector<float> x(k);
        for (float & v : x) {
            v = x_dist(rng);
        }

        std::vector<int8_t> q8_qs;
        std::vector<sycl::half> q8_d;
        std::vector<sycl::half> q8_s;
        quantize_q8_1_host(x, q8_qs, q8_d, q8_s);

        std::vector<uint8_t> q4_qs(static_cast<size_t>(n) * (k / 2));
        std::vector<sycl::half> q4_d(static_cast<size_t>(n) * nb);
        for (uint8_t & v : q4_qs) {
            v = static_cast<uint8_t>(q4_dist(rng));
        }
        for (sycl::half & d : q4_d) {
            d = sycl::half(d_dist(rng));
        }

        std::vector<float> ref;
        const auto ref0 = std::chrono::steady_clock::now();
        cpu_ref_q4_0_q8_1(q4_qs, q4_d, q8_qs, q8_d, q8_s, n, k, ref);
        const auto ref1 = std::chrono::steady_clock::now();
        const double ref_ms = std::chrono::duration<double, std::milli>(ref1 - ref0).count();

        sycl::queue q(sycl::gpu_selector_v);
        std::cout << "device=" << q.get_device().get_info<sycl::info::device::name>() << "\n";

        uint8_t * d_q4_qs = sycl::malloc_device<uint8_t>(q4_qs.size(), q);
        sycl::half * d_q4_d = sycl::malloc_device<sycl::half>(q4_d.size(), q);
        int8_t * d_q8_qs = sycl::malloc_device<int8_t>(q8_qs.size(), q);
        sycl::half * d_q8_d = sycl::malloc_device<sycl::half>(q8_d.size(), q);
        sycl::half * d_q8_s = sycl::malloc_device<sycl::half>(q8_s.size(), q);
        float * d_out = sycl::malloc_device<float>(n, q);
        if (!d_q4_qs || !d_q4_d || !d_q8_qs || !d_q8_d || !d_q8_s || !d_out) {
            throw std::runtime_error("device allocation failed");
        }

        q.memcpy(d_q4_qs, q4_qs.data(), q4_qs.size() * sizeof(uint8_t));
        q.memcpy(d_q4_d, q4_d.data(), q4_d.size() * sizeof(sycl::half));
        q.memcpy(d_q8_qs, q8_qs.data(), q8_qs.size() * sizeof(int8_t));
        q.memcpy(d_q8_d, q8_d.data(), q8_d.size() * sizeof(sycl::half));
        q.memcpy(d_q8_s, q8_s.data(), q8_s.size() * sizeof(sycl::half));
        q.memset(d_out, 0, n * sizeof(float));
        q.wait();

        for (int i = 0; i < args.warmup; ++i) {
            run_esimd_ks(q, d_q4_qs, d_q4_d, d_q8_qs, d_q8_d, d_q8_s, d_out, n, k, args.ks);
        }
        q.wait();

        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < args.iters; ++i) {
            run_esimd_ks(q, d_q4_qs, d_q4_d, d_q8_qs, d_q8_d, d_q8_s, d_out, n, k, args.ks);
        }
        q.wait();
        const auto t1 = std::chrono::steady_clock::now();

        std::vector<float> got(n);
        q.memcpy(got.data(), d_out, n * sizeof(float)).wait();

        double max_abs = 0.0;
        double max_rel = 0.0;
        double rms = 0.0;
        for (int i = 0; i < n; ++i) {
            const double abs_err = std::fabs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
            const double denom = std::max(1e-6, std::fabs(static_cast<double>(ref[i])));
            max_abs = std::max(max_abs, abs_err);
            max_rel = std::max(max_rel, abs_err / denom);
            rms += abs_err * abs_err;
        }
        rms = std::sqrt(rms / n);

        const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double avg_us = total_ms * 1000.0 / args.iters;
        const double weight_gb = (static_cast<double>(n) * (k / 2) + static_cast<double>(n) * nb * sizeof(sycl::half)) / 1.0e9;
        const double q8_gb = (static_cast<double>(k) + static_cast<double>(nb) * 2.0 * sizeof(sycl::half)) / 1.0e9;
        const double out_gb = static_cast<double>(n) * sizeof(float) / 1.0e9;
        const double gbps = (weight_gb + q8_gb + out_gb) / (avg_us * 1.0e-6);

        std::cout << std::fixed << std::setprecision(6)
                  << "n=" << n << " k=" << k << " ks=" << args.ks << " iters=" << args.iters << " warmup=" << args.warmup << "\n"
                  << "cpu_ref_ms=" << ref_ms << "\n"
                  << "esimd_avg_us=" << avg_us << "\n"
                  << "approx_gbps=" << gbps << "\n"
                  << "max_abs=" << max_abs << " max_rel=" << max_rel << " rms_abs=" << rms << "\n";

        sycl::free(d_q4_qs, q);
        sycl::free(d_q4_d, q);
        sycl::free(d_q8_qs, q);
        sycl::free(d_q8_d, q);
        sycl::free(d_q8_s, q);
        sycl::free(d_out, q);
        return (max_abs < 1e-3 || max_rel < 1e-4) ? 0 : 2;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
