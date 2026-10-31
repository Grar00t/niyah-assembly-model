# Morphology and tensor compiler bridge

`Makefile.ir` invokes the external niyah-lang `niyahc-tile` compiler, passes its live intervals to this repository's allocator, then assembles and links emitted code with `src/tensor_runtime.s` and `src/tensor_dispatch.s`. The compiler source is not included here. Vendored morphology origins are recorded in `src/morph_tokenizer/upstream/PROVENANCE.json`.

```sh
make NIYAH_LANG=/absolute/path/to/niyah-lang ir-demo
make NIYAH_LANG=/absolute/path/to/niyah-lang test-ir-bridge
./build/ir_bridge/demo.elf features
```

The `NIYAH_LANG` path must contain the compiler's `tensor-backend` target. The Makefile's sibling-directory default is a local development convention, not an included dependency.

The pipeline emits canonical IR, interval records, tile assignment and Assembly. The bridge validates returned colors against reserved resources and overlapping intervals. The runtime provides standard integer GEMM and packing. CPU/OS checks select scalar, AVX2, AVX512 or AMX; unsupported explicit modes are rejected.

The tests specify signed INT8 matrices of shapes 16-by-64 and 64-by-16, with INT32 output of shape 16-by-16. Static source/test excerpts are in [`evidence/docs-refresh-20261031/contract-excerpts.json`](../evidence/docs-refresh-20261031/contract-excerpts.json). Those dimensions are contracts, not new execution measurements.

No completed bridge receipt is included in the inspected evidence directories. Tests exist in `tests/test_ir_bridge.py` and `tests/test_dispatch_contract.py`; their coverage is not a substitute for an executed result. [`evidence/docs-refresh-20261031/inspection.json`](../evidence/docs-refresh-20261031/inspection.json)

The bridge requires conservative liveness and fixed tensor shapes. It does not compile arbitrary natural-language requests or integrate tensors with language-model generation. Linux xstate and Intel architecture references are retained in [SOURCES.md](SOURCES.md).
