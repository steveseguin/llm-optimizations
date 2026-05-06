#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

static void usage(const char * argv0) {
    std::fprintf(stderr,
            "usage: %s -m model.gguf [-dev SYCL0,SYCL1] [-sm none|layer|row|tensor] [-ts 1,1,1] "
            "[-ngl 99] [-c ctx] [-b batch] [-ub ubatch] [-t threads] [-n predict] [-p prompt]\n",
            argv0);
}

static std::vector<std::string> split(const std::string & s, char delim) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        const size_t pos = s.find(delim, start);
        out.push_back(s.substr(start, pos == std::string::npos ? std::string::npos : pos - start));
        if (pos == std::string::npos) {
            break;
        }
        start = pos + 1;
    }
    return out;
}

static uint64_t fnv1a_logits(const float * logits, int n_vocab) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < n_vocab; ++i) {
        uint32_t bits;
        static_assert(sizeof(bits) == sizeof(logits[i]));
        std::memcpy(&bits, &logits[i], sizeof(bits));
        for (int b = 0; b < 4; ++b) {
            h ^= (bits >> (8 * b)) & 0xffu;
            h *= 1099511628211ull;
        }
    }
    return h;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "Deterministic token probe.";
    std::string dev_arg;
    std::string sm_arg = "none";
    std::string ts_arg;
    int ngl = 99;
    int n_ctx = 2048;
    int n_batch = 2048;
    int n_ubatch = 32;
    int n_threads = 8;
    int n_predict = 16;

    for (int i = 1; i < argc; ++i) {
        auto need_value = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                usage(argv[0]);
                std::exit(2);
            }
            return argv[++i];
        };

        if (std::strcmp(argv[i], "-m") == 0) {
            model_path = need_value("-m");
        } else if (std::strcmp(argv[i], "-p") == 0) {
            prompt = need_value("-p");
        } else if (std::strcmp(argv[i], "-dev") == 0 || std::strcmp(argv[i], "--device") == 0) {
            dev_arg = need_value("-dev");
        } else if (std::strcmp(argv[i], "-sm") == 0 || std::strcmp(argv[i], "--split-mode") == 0) {
            sm_arg = need_value("-sm");
        } else if (std::strcmp(argv[i], "-ts") == 0 || std::strcmp(argv[i], "--tensor-split") == 0) {
            ts_arg = need_value("-ts");
        } else if (std::strcmp(argv[i], "-ngl") == 0) {
            ngl = std::atoi(need_value("-ngl"));
        } else if (std::strcmp(argv[i], "-c") == 0) {
            n_ctx = std::atoi(need_value("-c"));
        } else if (std::strcmp(argv[i], "-b") == 0) {
            n_batch = std::atoi(need_value("-b"));
        } else if (std::strcmp(argv[i], "-ub") == 0) {
            n_ubatch = std::atoi(need_value("-ub"));
        } else if (std::strcmp(argv[i], "-t") == 0) {
            n_threads = std::atoi(need_value("-t"));
        } else if (std::strcmp(argv[i], "-n") == 0) {
            n_predict = std::atoi(need_value("-n"));
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    if (model_path.empty()) {
        usage(argv[0]);
        return 2;
    }

    ggml_backend_load_all();

    std::vector<ggml_backend_dev_t> devices;
    if (!dev_arg.empty() && dev_arg != "none") {
        for (const auto & name : split(dev_arg, ',')) {
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str());
            if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                std::fprintf(stderr, "invalid device: %s\n", name.c_str());
                return 2;
            }
            devices.push_back(dev);
        }
        devices.push_back(nullptr);
    }

    std::vector<float> tensor_split(llama_max_devices(), 0.0f);
    if (!ts_arg.empty()) {
        const auto parts = split(ts_arg, ',');
        if (parts.size() > tensor_split.size()) {
            std::fprintf(stderr, "too many tensor split entries\n");
            return 2;
        }
        for (size_t i = 0; i < parts.size(); ++i) {
            tensor_split[i] = std::stof(parts[i]);
        }
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    if (!devices.empty()) {
        mparams.devices = devices.data();
    }
    if (sm_arg == "none") {
        mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
    } else if (sm_arg == "layer") {
        mparams.split_mode = LLAMA_SPLIT_MODE_LAYER;
    } else if (sm_arg == "row") {
        mparams.split_mode = LLAMA_SPLIT_MODE_ROW;
    } else if (sm_arg == "tensor") {
        mparams.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    } else {
        std::fprintf(stderr, "invalid split mode: %s\n", sm_arg.c_str());
        return 2;
    }
    if (!ts_arg.empty()) {
        mparams.tensor_split = tensor_split.data();
    }

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        std::fprintf(stderr, "failed to size prompt tokens\n");
        return 1;
    }

    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        std::fprintf(stderr, "failed to tokenize prompt\n");
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_batch;
    cparams.n_ubatch = n_ubatch;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.type_k = GGML_TYPE_F16;
    cparams.type_v = GGML_TYPE_F16;
    cparams.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    std::fprintf(stdout, "{\"event\":\"prompt\",\"n_prompt\":%d,\"n_predict\":%d}\n", n_prompt, n_predict);
    std::fflush(stdout);

    for (int step = 0; step < n_predict; ++step) {
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "llama_decode failed at step %d\n", step);
            break;
        }

        const float * logits = llama_get_logits_ith(ctx, -1);
        if (logits == nullptr) {
            std::fprintf(stderr, "missing logits at step %d\n", step);
            break;
        }

        std::vector<int> top;
        top.reserve(5);
        for (int i = 0; i < n_vocab; ++i) {
            if ((int) top.size() < 5) {
                top.push_back(i);
                continue;
            }
            int min_pos = 0;
            for (int j = 1; j < 5; ++j) {
                if (logits[top[j]] < logits[top[min_pos]]) {
                    min_pos = j;
                }
            }
            if (logits[i] > logits[top[min_pos]]) {
                top[min_pos] = i;
            }
        }
        std::sort(top.begin(), top.end(), [&](int a, int b) {
            if (logits[a] == logits[b]) {
                return a < b;
            }
            return logits[a] > logits[b];
        });

        llama_token token = (llama_token) top[0];
        const uint64_t hash = fnv1a_logits(logits, n_vocab);
        std::fprintf(stdout, "{\"step\":%d,\"token\":%d,\"logit_hash\":\"%016" PRIx64 "\",\"top\":[", step, token, hash);
        for (size_t i = 0; i < top.size(); ++i) {
            std::fprintf(stdout, "%s{\"id\":%d,\"logit\":%.9g}", i ? "," : "", top[i], logits[top[i]]);
        }
        std::fprintf(stdout, "]}\n");
        std::fflush(stdout);

        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }
        batch = llama_batch_get_one(&token, 1);
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
