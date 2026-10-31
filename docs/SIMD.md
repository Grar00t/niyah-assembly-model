# Byte-model SIMD and kernel evidence

The `dot16` implementation in `src/attention_lm.s` selects scalar SSE or AVX2/FMA after CPUID and XGETBV checks. `AVX2_FAST` bypasses runtime selection and is a test build option. Forced fallback tests exercise a branch; they are not execution on a processor lacking AVX2. [`evidence/docs-refresh-20261031/contract-excerpts.json`](../evidence/docs-refresh-20261031/contract-excerpts.json)

These are standard dot products and register blocking. Existing implementation references are NN_S (zenyyxz), matmul_assembly_x86 (alint77) and the Intel Optimization Reference Manual. URLs and license status are in [SOURCES.md](SOURCES.md).

## Recorded byte-model benchmark

The receipt records an Intel Core i5-10400F, a 93-byte fixture, 1200 epochs, 12 trials per path and 2 warmups. This is a historical workload, not the current corpus. [`evidence/simd-benchmark-local.json`](../evidence/simd-benchmark-local.json)

| Path | Median seconds | Scalar/path ratio | Evidence |
| --- | ---: | ---: | --- |
| Scalar | 0.999423 | 1 | [`evidence/simd-benchmark-local.json`](../evidence/simd-benchmark-local.json) |
| Auto AVX2/FMA | 0.792365 | 1.2613 | [`evidence/simd-benchmark-local.json`](../evidence/simd-benchmark-local.json) |
| Forced AVX2/FMA | 0.781844 | 1.2783 | [`evidence/simd-benchmark-local.json`](../evidence/simd-benchmark-local.json) |

All 36 timing samples remain in that receipt. The current byte-model regression is described in [ATTENTION.md](ATTENTION.md). Unsupported numerical comparison values formerly printed here were omitted; see [`evidence/docs-refresh-20261031/claims.json`](../evidence/docs-refresh-20261031/claims.json).

## Integer kernel suite

The serial GEMM/dot suite records 1002 verification cases PASS in [`evidence/kernel-suite-20261031-112031-843492/results-o2-full/verification.log`](../evidence/kernel-suite-20261031-112031-843492/results-o2-full/verification.log). Compiler options and input hashes are in [`evidence/kernel-suite-20261031-112031-843492/results-o2-full/metadata.json`](../evidence/kernel-suite-20261031-112031-843492/results-o2-full/metadata.json); all measured durations remain in [`evidence/kernel-suite-20261031-112031-843492/results-o2-full/benchmark.csv`](../evidence/kernel-suite-20261031-112031-843492/results-o2-full/benchmark.csv).

The integer kernel uses shared scratch. Output canaries and exact integer oracles cover the tested calls; they do not test concurrency or every possible read boundary. These kernel checks do not establish full-model quality.
