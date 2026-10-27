# Native dot16 SIMD and runtime dispatch: M3 learned-QKV experiment

## Kernel implementation and provenance

The hand-written `dot16` kernel is shared across the model's
**three learnable Q/K/V projections**, attention Q·K score, residual
feed-forward projection and output projection, all using 16-wide float32
vectors. The portable path executes SSE scalar multiply-add. The accelerated
path processes two 8-lane 256-bit vectors with AVX2/FMA3
(`vmulps`, `vfmadd231ps`) then horizontally sums with `vaddps`,
`vextractf128` and `vhaddps`. SSE and AVX2 have different floating-point
addition orders; bitwise identical weights are NOT a valid expectation after
240 stochastic-gradient-descent epochs.

Inspired by (and independently implemented, **no copied code**):
[NN_S](https://github.com/zenyyxz/NN_S) (MIT-licensed assembly NN),
[matmul_assembly_x86](https://github.com/alint77/matmul_assembly_x86), and
[Intel's Optimization Reference Manual](https://www.intel.com/content/www/us/en/content-details/671488/intel-64-and-ia-32-architectures-optimization-reference-manual-volume-1.html).
Third-party repositories' claimed GFLOPs were NOT assumed for this model.

## CPU guard

Normal native build: `make run-attention-auto` / `AVX2_AUTO=1`. At
startup, CPUID first verifies feature-leaf 7, then OSXSAVE, AVX,
FMA3; XGETBV(0) checks OS-enabled XMM/YMM state; CPUID(7).EBX.AVX2 is
checked before executing vector instructions. The program prints
`kernel_avx2 1` or `kernel_avx2 0`. The portable fallback remains available
even on older machines. `FORCE_SCALAR=1` in test builds forces fallback
to exercise its code path, but does NOT constitute physical testing on
an AVX2-incapable CPU. The unguarded `AVX2_FAST` compile flag is
testing-only: do not ship as a universal binary.

## Actual WSL hardware: 2026-10-15

CPU: Intel Core i5-10400F @ 2.90 GHz, Ubuntu under WSL;
93-byte training corpus, **1200 epochs**, 12 repetitions per kernel
after two warmups, rotated execution order between the three modes.
`tools/bench_avx2.py` uses a Python subprocess timer only; the trained
model runs in native assembly with no Python interpreter loaded into it.

| M3 kernel | Training wall-time median | Scalar / this |
| --- | ---: | ---: |
| Portable scalar | 0.999423 s | 1.0000x |
| CPUID-dispatched AVX2/FMA3 | 0.792365 s | **1.2613x** |
| Explicit unguarded AVX2/FMA3 | 0.781844 s | 1.2783x |

All 36 raw timing values in
[../evidence/simd-benchmark-local.json](../evidence/simd-benchmark-local.json).
The performance estimate is specific to the **9,616-parameter, 8-byte context**
changes can move these results.

## Independent numerical verification

Run `make test-simd`. A Python mathematical oracle independently
recomputes output probabilities and losses from the weights produced by
both Assembly binaries. It checks:

- Binary is statically linked and has no ELF interpreter.
- CPUID path chooses AVX2/FMA on the tested CPU.
- Forced fallback and scalar train to **exactly identical** weights.
- Auto and explicitly accelerated modes yield identical weights on this CPU.
- Each path passes the independent oracle for train and heldout loss.
- Deterministic repeats reproduce the same result on fixed seed.
- Independent context distribution and total losses agree within declared
  numeric tolerances.

Actual 240-epoch M3 comparisons:

| Numerical quantity | Measured |
| --- | ---: |
| Max scalar vs AVX2 trained weight absolute difference | 0.0678094774 |
| Maximum output-probability L1 difference over 3 contexts | 0.00168739895 |
| Scalar vs AVX2 train loss difference | 0.0182109632 milli-nats |
| Scalar vs AVX2 heldout loss difference | 0.0755226275 milli-nats |

The raw *weights* differ more than M2 because FP32 rounding accumulates
through independent trainable attention projections and a nonconvex
optimization trajectory. We **do not** hide this difference. The tests
compare model behavior through independently computed objective and
probabilities, not just a filename, hash or single lucky generation.

## Compatibility and safety

The original trigram tests, checkpoint exact replay, and four disposable
corruption/missing-file cases are in GitHub Actions. No Win32/MSVC
`cl.exe`, paid compilers, cloud GPU rentals or hidden dependencies were used.
The new neural raw weight format is **v2** and intentionally fails closed
on incompatible older v1 checkpoints.

or proof of invention outside learned parameters.
