# Native dot16: runtime-dispatched AVX2/FMA3 with a safe scalar path

## Motivation, sources, and licensing

The original attention LM used scalar SSE multiply-add loops for the 16-wide
query/key inner product, feed-forward matrix/vector multiplication and output
projection. This change adapts the **SIMD-first** engineering approach seen in
[NN_S](https://github.com/zenyyxz/NN_S) (MIT) and the optimization methodology
in [Intel's Optimization Reference Manual](https://www.intel.com/content/www/us/en/content-details/671488/intel-64-and-ia-32-architectures-optimization-reference-manual-volume-1.html).
Another related reference is
[matmul_assembly_x86](https://github.com/alint77/matmul_assembly_x86).

**No source code copied from those projects.** The new GNU-as Intel-syntax
routine was written for this repository. Those references are learning aids,
not authorities for this model's accuracy or speed.

## Executed kernels

There is one `dot16` kernel used for three real operations:

1. One-head attention score `Q dot K[t]`.
2. Residual feed-forward `F[row] dot H`.
3. Output projection `O[token] dot Y`.

- The scalar path uses 16 `mulss` / `addss` operations.
- The AVX2/FMA3 path multiplies eight float32 lanes in parallel using
  `vmulps`, fuses the next eight multiply-adds with `vfmadd231ps`,
  and reduces the 8 lanes with `vextractf128`, `vaddps` and `vhaddps`.
- A *single* kernel code path is invoked from each operation; no C or Python
  functions are called by the resulting model binary.
- Different addition association and FMA rounding mean the scalar and vector
  FP32 weights are not bit-for-bit identical after SGD. We test probability
  and loss equivalence within measured bounds, not arbitrary hash equality.

### CPU safety and fallback

`-Wa,--defsym,AVX2_AUTO=1` compiles both paths. Native x86-64 `CPUID`
tests OSXSAVE, AVX, FMA and AVX2; `XGETBV(0)` verifies that the OS enables
both XMM and YMM state before the AVX2 branch may execute. On unsupported
machines, execution remains on the x86-64 scalar path. The program prints
`kernel_avx2 0` or `kernel_avx2 1` when auto-dispatch is enabled.

`-Wa,--defsym,AVX2_FAST=1` compiles an **unconditionally accelerated**
variant and is unsafe on unsupported CPUs. Use `AVX2_AUTO` in production.

`-Wa,--defsym,FORCE_SCALAR=1` together with AVX2_AUTO deliberately
forces the compatibility path for regression tests, even on a compatible CPU.
This proves the fallback implementation executes, but it is *not* a physical
test on a CPU lacking AVX2. CPUID branches and OS feature combinations still
need validation on heterogeneous real hardware.

## Local verification receipts (2026-10-15, DESKTOP-2P1JAIV Ubuntu/WSL)

```sh
make run-attention-auto
make test-attention
make test-simd
make attention-checkpoint verify-attention test-attention-checkpoint
make test-attention-negative
make generate-attention-auto
make bench-avx2
```

Test checks independently recompute all masked next-byte probabilities from
the final Assembly-trained weights and compare on both implementations.
The runtime-dispatch forced-scalar trained weights match the scalar binary
**exactly**. The AVX2 AUTO result equals AVX2 FAST results on the compatible
test machine, and deterministic runs yield the same weights for a fixed seed.

The local measured max weight absolute difference (scalar vs vector)
after 240 epochs was **0.000413802**, and L1 probability difference on
`the cat ` was **0.000070483**. The final training and related heldout
negative log losses agree when rounded to milli-nats (0.174 and 0.240).

### Practical speed (bounded microbenchmark)

At 1,200 epochs, 12 **order-rotated**, wall-clock repetitions after warm-up,
the median times on this Intel i5-10400F (Ubuntu under WSL) were:

| Kernel | Median training time | Relative |
| --- | ---: | ---: |
| scalar fallback | 0.395372 s | 1.0x |
| AUTO: CPUID-dispatched AVX2/FMA3 | 0.363059 s | 1.089x faster |
| explicitly enabled AVX2/FMA3 | 0.362985 s | 1.089x faster |

The result is an *observed microbenchmark*, not guaranteed speed for other
CPUs, datasets or larger Transformer layers. OS scheduling, WSL behavior,
thermal state and clock frequency affect timing. Full raw durations and
test parameters are in `evidence/simd-benchmark-local.json`.

### Fail-closed model persistence

The separately generated checkpoint remains 35,392 bytes and is verified
against SHA-256 of source, training/heldout files and trained weights before
inference. A test mutates *disposable copies* of a parameter, truncates the
weight file, changes the source digest and removes the weights, and confirms
all four cases are rejected. The ordinary scalar exact checkpoint replay
test is also retained.

### What this does NOT mean

This optimization accelerates the *same* tiny 8,848-parameter, 8-byte-context
single-head educational byte predictor. It does not add learned Q/K/V
matrices, multiple decoder blocks, a tokenizer, instruction following,
reasoning or semantic generalization. It is not a general-purpose AVX2 GEMM,
a benchmark against OpenBLAS, or evidence of an LLM-scale speedup.

Do not invoke arbitrary cloud/health/finance connectors or incur GPU costs
to perform native CPU kernel microbenchmarks.
