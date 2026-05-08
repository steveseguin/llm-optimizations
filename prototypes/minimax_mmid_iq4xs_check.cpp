// Minimal GGML MUL_MAT_ID IQ4_XS correctness probe for the MiniMax fast expert-down path.
//
// Build example:
//   source /opt/intel/oneapi/setvars.sh
//   icpx -std=c++17 -O2 minimax_mmid_iq4xs_check.cpp \
//     -I/home/steve/src/ik_llama.cpp/ggml/include \
//     -I/home/steve/src/ik_llama.cpp/ggml/src \
//     -L/home/steve/src/ik_llama.cpp/build-sycl-rpc-b70/ggml/src -lggml \
//     -Wl,-rpath,/home/steve/src/ik_llama.cpp/build-sycl-rpc-b70/ggml/src \
//     -o /tmp/minimax_mmid_iq4xs_check

#include <ggml.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct graph_case {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * as = nullptr;
    ggml_tensor * b = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * out = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
};

static graph_case make_case(ggml_backend_t backend, ggml_type type_a, int n_mats, int n_used, int n_tokens, int m, int k) {
    size_t ctx_size = 64 * 1024 * 1024;
    ggml_init_params params = {ctx_size, nullptr, true};
    graph_case gc;
    gc.ctx = ggml_init(params);
    if (!gc.ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        std::exit(2);
    }

    gc.as = ggml_new_tensor_3d(gc.ctx, type_a, k, m, n_mats);
    gc.ids = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_I32, n_used, n_tokens);
    gc.b = ggml_new_tensor_3d(gc.ctx, GGML_TYPE_F32, k, n_used, n_tokens);
    gc.out = ggml_mul_mat_id(gc.ctx, gc.as, gc.b, gc.ids);

    gc.graph = ggml_new_graph(gc.ctx);
    ggml_build_forward_expand(gc.graph, gc.out);

    gc.buffer = ggml_backend_alloc_ctx_tensors(gc.ctx, backend);
    if (!gc.buffer) {
        std::fprintf(stderr, "alloc tensors failed for backend %s\n", ggml_backend_name(backend));
        std::exit(3);
    }
    return gc;
}

static void free_case(graph_case & gc) {
    if (gc.buffer) {
        ggml_backend_buffer_free(gc.buffer);
    }
    if (gc.ctx) {
        ggml_free(gc.ctx);
    }
}

static void fill_case(const graph_case & gc, const std::vector<uint8_t> & qweights,
                      const std::vector<float> & b, const std::vector<int32_t> & ids) {
    ggml_backend_tensor_set(gc.as, qweights.data(), 0, qweights.size());
    ggml_backend_tensor_set(gc.b, b.data(), 0, b.size() * sizeof(float));
    ggml_backend_tensor_set(gc.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
}

static std::vector<float> read_output(const graph_case & gc) {
    std::vector<float> out(ggml_nelements(gc.out));
    ggml_backend_tensor_get(gc.out, out.data(), 0, out.size() * sizeof(float));
    return out;
}

struct diff_stats {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double nmse = 0.0;
    double sum = 0.0;
    double sumsq = 0.0;
};

static diff_stats compare_outputs(const std::vector<float> & ref, const std::vector<float> & got) {
    diff_stats stats;
    double se = 0.0;
    double norm = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double a = ref[i];
        const double b = got[i];
        const double d = a - b;
        se += d * d;
        norm += a * a;
        stats.sum += b;
        stats.sumsq += b * b;
        stats.max_abs = std::max(stats.max_abs, std::abs(d));
        stats.max_rel = std::max(stats.max_rel, std::abs(d) / std::max(1e-9, std::abs(a)));
    }
    stats.nmse = se / std::max(norm, 1e-30);
    return stats;
}

static std::vector<float> reference_mmid_iq4_xs(const std::vector<uint8_t> & qweights,
                                                const std::vector<float> & b,
                                                const std::vector<int32_t> & ids,
                                                int n_mats, int n_used, int n_tokens, int m, int k) {
    (void)n_mats;
    const ggml_type_traits_t traits = ggml_internal_get_type_traits(GGML_TYPE_IQ4_XS);
    if (!traits.to_float) {
        std::fprintf(stderr, "IQ4_XS to_float unavailable\n");
        std::exit(4);
    }

    const size_t row_size = ggml_row_size(GGML_TYPE_IQ4_XS, k);
    std::vector<float> row((size_t)k);
    std::vector<float> out((size_t)m * n_used * n_tokens);

    for (int t = 0; t < n_tokens; ++t) {
        for (int e = 0; e < n_used; ++e) {
            const int expert = ids[(size_t)t * n_used + e];
            const float * vec = b.data() + (size_t)k * (e + n_used * t);
            for (int y = 0; y < m; ++y) {
                const uint8_t * qrow = qweights.data() + row_size * ((size_t)expert * m + y);
                traits.to_float(qrow, row.data(), k);
                double acc = 0.0;
                for (int x = 0; x < k; ++x) {
                    acc += (double)row[x] * vec[x];
                }
                out[(size_t)y + (size_t)m * (e + n_used * t)] = (float)acc;
            }
        }
    }

    return out;
}

int main() {
    const ggml_type type_a = GGML_TYPE_IQ4_XS;
    const int n_mats = 8;
    const int n_used = 4;
    const int n_tokens = 4;
    const int m = 512;
    const int k = 256;

    ggml_backend_t sycl = nullptr;
    for (size_t i = 0; i < ggml_backend_reg_get_count(); ++i) {
        const char * name = ggml_backend_reg_get_name(i);
        if (std::strstr(name, "SYCL") != nullptr) {
            sycl = ggml_backend_reg_init_backend(i, nullptr);
            break;
        }
    }
    if (!sycl) {
        std::fprintf(stderr, "SYCL backend not found; registered backends=%zu\n", ggml_backend_reg_get_count());
        return 2;
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> weights_f32((size_t)n_mats * m * k);
    for (float & v : weights_f32) {
        v = dist(rng);
    }

    const size_t qbytes = ggml_row_size(type_a, k) * (size_t)n_mats * m;
    std::vector<uint8_t> weights_q(qbytes);
    std::vector<float> imatrix(k, 1.0f);
    ggml_quantize_chunk(type_a, weights_f32.data(), weights_q.data(), 0, (int64_t)n_mats * m, k, imatrix.data(), nullptr);

    std::vector<float> b((size_t)k * n_used * n_tokens);
    for (float & v : b) {
        v = dist(rng);
    }

    std::vector<int32_t> ids((size_t)n_used * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        for (int i = 0; i < n_used; ++i) {
            ids[(size_t)t * n_used + i] = (i * 3 + t) % n_mats;
        }
    }

    graph_case cpu_case = make_case(cpu, type_a, n_mats, n_used, n_tokens, m, k);
    graph_case sycl_case = make_case(sycl, type_a, n_mats, n_used, n_tokens, m, k);
    fill_case(cpu_case, weights_q, b, ids);
    fill_case(sycl_case, weights_q, b, ids);

    if (ggml_backend_graph_compute(cpu, cpu_case.graph) != GGML_STATUS_SUCCESS ||
        ggml_backend_graph_compute(sycl, sycl_case.graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed\n");
        return 3;
    }
    ggml_backend_synchronize(cpu);
    ggml_backend_synchronize(sycl);

    const auto cpu_out = read_output(cpu_case);
    const auto sycl_out = read_output(sycl_case);
    const auto ref_out = reference_mmid_iq4_xs(weights_q, b, ids, n_mats, n_used, n_tokens, m, k);
    const diff_stats cpu_vs_ref = compare_outputs(ref_out, cpu_out);
    const diff_stats sycl_vs_ref = compare_outputs(ref_out, sycl_out);
    const diff_stats sycl_vs_cpu = compare_outputs(cpu_out, sycl_out);

    std::printf("backend=%s elements=%zu cpu_ref_max_abs=%g cpu_ref_max_rel=%g cpu_ref_nmse=%g "
                "sycl_ref_max_abs=%g sycl_ref_max_rel=%g sycl_ref_nmse=%g "
                "sycl_cpu_max_abs=%g sycl_cpu_max_rel=%g sycl_cpu_nmse=%g "
                "sycl_sum=%.17g sycl_sumsq=%.17g first=",
                ggml_backend_name(sycl), cpu_out.size(),
                cpu_vs_ref.max_abs, cpu_vs_ref.max_rel, cpu_vs_ref.nmse,
                sycl_vs_ref.max_abs, sycl_vs_ref.max_rel, sycl_vs_ref.nmse,
                sycl_vs_cpu.max_abs, sycl_vs_cpu.max_rel, sycl_vs_cpu.nmse,
                sycl_vs_ref.sum, sycl_vs_ref.sumsq);
    for (size_t i = 0; i < std::min<size_t>(8, sycl_out.size()); ++i) {
        std::printf("%s%.9g", i == 0 ? "" : ",", sycl_out[i]);
    }
    std::printf("\n");

    free_case(cpu_case);
    free_case(sycl_case);
    ggml_backend_free(cpu);
    ggml_backend_free(sycl);

    return sycl_vs_ref.nmse < 5e-4 ? 0 : 1;
}
