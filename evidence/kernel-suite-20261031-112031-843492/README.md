# Niyah kernel verification and benchmark suite

Linux x86-64, GCC/Clang. No Python ML packages, model weights, network, or repository changes.

Place the two source files together. Run:

    python3 run_suite.py --repo "$HOME/niyah-assembly-model" --out /tmp/niyah-kernels-o2 --opt O2
    python3 run_suite.py --repo "$HOME/niyah-assembly-model" --out /tmp/niyah-kernels-o3 --opt O3 --full

Output directories must not already exist. --verify-only skips timing.
--full adds complete-output verification and benchmarks for representative Qwen linear shapes.
The runner extracts literal ASM from tools/qwen/qwen_native_layer0.py with ast.literal_eval;
it never imports that file. It links src/niyah_i8_gemm.s without changing it.

Outputs: metadata.json (compiler, source hashes, flags, exit code), verification.log,
benchmark.csv. A failed verification exits nonzero before any timing. AVX2 absence exits 77.
Do not compile the harness with -mavx2/-march=native: baseline dispatch must run safely first.
The supported builds disable fast-math, FMA contraction and compiler loop vectorization;
the external assembly still uses its own vector instructions. Compare optimization levels
on the same host. For affinity, use taskset -c <an-allowed-cpu> before python3.

Coverage:
- 12 invalid geometry cases, including signed-int32 product overflow; output unchanged.
- 336 fixed GEMM cases (12 token sizes x 7 K values x 4 value patterns).
- 128 reproducible random GEMM cases; exact independent INT64 oracle for every output.
- Output byte canaries, deliberately SIMD-unaligned inputs, input immutability,
  repeated-call determinism, K=16384 boundary, 8/4/1 panel transitions.
- 514 dot128 cases: deterministic random, zero, cancellation, sparse, tiny/large finite,
  extreme float products, NaN and infinity. Long-double oracle with magnitude-based
  rounding tolerance; assumes normal round-to-nearest and default FTZ/DAZ settings.
- Default total: 990 verification cases; --full adds 12 full-size GEMM cases.
- 9 timed samples with a warm-up and adaptive GEMM batching, fixed dot batching;
  min/median/max microseconds and nominal Gops/s (multiply+add counted as two).
  Includes call overhead. Resident-buffer timings, not a cold-cache benchmark.

Limits:
- GEMM uses static ai16 scratch: run serially, no concurrency claims.
- No ABI register-preservation probe, guard-page overread test, or exhaustive memory proof.
  Canaries detect tested writes, not reads. Assembly is not instrumented by C sanitizers.
- No invalid-pointer tests: null-pointer rejection is not part of the supplied kernel contract.
- No model-quality, fake-quant, g32 scale, or full-inference validation.
- Not an optimization of the assembly, and no claim of speedup over PyTorch.

Author-side validation:
- C suite compiled as an object with GCC -std=c11 -O2 -Wall -Wextra -Werror
  -fno-fast-math -ffp-contract=off.
- Actual niyah_dot128 ASM extracted from the supplied paste executed: 514 cases PASS.
- Full GEMM suite and performance measurements are NOT executed in the author sandbox.
  Run locally to produce them against your checked-out assembly.
