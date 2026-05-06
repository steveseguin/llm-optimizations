# 2026-05-06 Q4 runtime MMVQ subgroup sweep

## Summary

Added a guarded runtime selector for the Q4_0 reordered MMVQ subgroup count:

```bash
GGML_SYCL_REORDER_MMVQ_SUBGROUPS_RUNTIME=1|2|4|8|16
```

When unset, the path uses the existing compiled default
`GGML_SYCL_REORDER_MMVQ_SUBGROUPS`, so the recommended launch recipe is unchanged.

This was motivated by the corrected standalone Q4_0 x Q8_1 harness, where some
hot shapes were faster with fewer workgroup subgroups than the default. In full
llama.cpp decode, the benefit did not carry through.

## Implementation

Changed `ggml/src/ggml-sycl/mmvq.cpp`:

- factored `reorder_mul_mat_vec_q4_0_q8_1_sycl` into templated subgroup variants;
- applied the same runtime selector to `reorder_mul_mat_vec_q4_0_q8_1_fused2_sycl`;
- left default behavior unchanged when the env var is not set.

Build command:

```bash
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
cmake --build /home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31 --target llama-bench -j 2
```

Patch artifact:

- Focused runtime-subgroup patch:
  `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-q4-runtime-subgroups-focused-20260506.patch`
- Focused SHA256: `299192c3ba4c6d86cbdd9f2def54947cac16bb6f9812ec52f55bd0234fe44cf2`
- `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-q4-runtime-subgroups-current-20260506.patch`
- SHA256: `19ed486971012f9d695c2e5946db99c6fe7a9ccfc11b5bd3cacc4305352fa543`
- Gzip: `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-q4-runtime-subgroups-current-20260506.patch.gz`
- Gzip SHA256: `a9d4135f4c997604745f6bfbf1605bc291f411b2b93c711239b7a3141d821dc7`

## Results

Single B70, p0/n128/r2, selector `level_zero:2`:

| subgroup override | decode tok/s |
| --- | ---: |
| default | 24.676064 |
| 1 | 24.658384 |
| 2 | 24.662854 |
| 4 | 24.640353 |
| 8 | 24.708439 |
| 16 | 24.808761 |

3x B70, p0/n128/r2, selector order `level_zero:2,1,3`:

| subgroup override | decode tok/s |
| --- | ---: |
| default | 44.882959 |
| 8 | 44.444759 |
| 16 | 45.248504 |

3x B70 full p512/n512/r3, selector order `level_zero:2,1,3`:

| subgroup override | prompt tok/s | decode tok/s |
| --- | ---: | ---: |
| 16 | 118.426089 | 45.411030 |
| default | 118.146716 | 45.932216 |

The current best validated Q4_0 3x result remains the earlier
`GGML_SYCL_COMM_SYNC_AFTER=2` run at `46.194319 tok/s`.

## Conclusion

This is a useful diagnostic knob but not a recommended performance setting.
The corrected microbenchmark still matters for future kernel work, but simple
subgroup-count tuning is too small once the whole decode graph, fused2 path,
launch overhead, linear attention, and collectives are included.

Do not submit these subgroup runs to LocalMaxxing; they do not beat the current
published Q4_0 result and do not represent a better user recipe.
