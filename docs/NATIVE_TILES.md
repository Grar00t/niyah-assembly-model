# Matrix kernels and dispatch

`src/tile_probe.s` and `src/tile_kernels.s` build a static Linux executable. `features` emits its capability record. Explicit scalar, AVX2, AVX512 and AMX modes pass through the applicable CPU/OS checks. An unsupported mode exits before executing its gated instructions. Contracts are recorded in [`evidence/docs-refresh-20261031/inspection.json`](../evidence/docs-refresh-20261031/inspection.json) and [`evidence/docs-refresh-20261031/contract-excerpts.json`](../evidence/docs-refresh-20261031/contract-excerpts.json).

```sh
make native-kernels test-kernels
./build/tile_probe features
```

The matrix operations use standard blocking. The FP32 probe multiplies 16-by-16 matrices; its AVX2 kernel reuses an input vector across 4 output rows. The AMX integer path packs its input for 16-by-64 times 64-by-16 multiplication. [`evidence/docs-refresh-20261031/contract-excerpts.json`](../evidence/docs-refresh-20261031/contract-excerpts.json)

`TILED_PROJECTIONS` opts the byte-model Q/K/V projection into register blocking. The historical receipt records byte-identical training weights and output for the compared paths; it does not establish compatibility after the current byte-model changes. [`evidence/nexus-kernels.json`](../evidence/nexus-kernels.json)

The recorded host executed scalar and AVX2 paths. AMX and AVX512 were not executed. The receipt contains 35 matrix fixtures and all command results. [`evidence/nexus-kernels.json`](../evidence/nexus-kernels.json)

Raw samples, medians, MAD, CPU affinity and source hashes remain in [`evidence/nexus-native-benchmark.json`](../evidence/nexus-native-benchmark.json). These are fixed-shape resident-workload measurements. [Allocator](TILE_ALLOCATOR.md), [compiler bridge](TENSOR_IR.md), [sources](SOURCES.md).
