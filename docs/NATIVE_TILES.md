# Native matrix and projection milestone

The executable is a freestanding Linux x86-64 ELF using direct kernel syscalls.
It has no libc, Python, PyTorch, CUDA, dynamic loader, or shared-library dependency
at runtime. GNU assembler/linker and Python's standard library remain build/test
tools. It is not a bootable bare-metal operating system.

## Executable contract

`build/tile_probe features` emits a 64-byte little-endian record: magic
`NYCPUID1`, usable capability bits, CPUID1.ECX, CPUID7.EBX, CPUID7.EDX,
XCR0, maximum basic leaf, reserved. Bit0 = AVX2/FMA/XMM+YMM enabled;
bit1 = AVX512F/FMA/XMM+YMM+ZMM enabled; bit2 = AMX-TILE+INT8 with
tile XCR0 state. Bit2 does not imply that Linux process permission exists.

`tile_probe scalar|auto|avx2|avx512 [repeat]` reads exactly 2048 bytes:
two row-major FP32 16x16 matrices A and B. It writes 1024 bytes of C=A*B.
Inputs must be finite. The arithmetic can still overflow for large finite
inputs; use the stated numerical domain and an application-level range gate.
No aliasing applies to the internal buffers. Repeat is in 1..10000000.

The AVX2 kernel holds four rows by eight output columns in registers and
reuses each B vector across four FMA chains. The AVX512F kernel holds four
rows by sixteen columns. Scalar uses non-fused multiply/add, so arbitrary
FP32 inputs need a rounding tolerance in the independent oracle.

`tile_probe amx-int8 [repeat]` reads signed INT8 A16x64 and B64x16
(row-major, 1024 bytes each); it packs B into K groups of four and emits
INT32 C16x16 (1024 bytes). It queries CPUID/XCR0, then Linux
ARCH_GET_XCOMP_SUPP and ARCH_REQ_XCOMP_PERM(18), before tile instructions.
The AMX path is assembled but NOT hardware-verified on the i5-10400F.
AVX512F is likewise not executable on that machine.

Exit0 success,64 invalid CLI,65 malformed input,74 output failure,77
unsupported ISA/OS permission. Unsupported requests return before reading
payload or executing the gated instruction. Native unit and differential
checks live in tests/test_tile_kernels.py.

## Existing model integration

`build/attention_lm_tiled` opts into TILED_PROJECTIONS with guarded AVX2.
Its Q/K/V projection computes four output rows together and reuses the
input vector. Reduction order within each row matches dot16; 240-epoch
training weights and stdout are required to remain byte-identical to
the same original scalar/AVX2 path. Existing default binaries/checkpoint
layouts remain compatible because arithmetic and parameter meaning are
unchanged. This is register blocking for matrix-vector projections;
it is distinct from AMX hardware tile registers and the GEMM experiment.

## Evidence boundaries

Local receipts: evidence/nexus-kernels.json and
evidence/nexus-native-benchmark.json. Benchmark warmed twice, alternates
order, pins one logical CPU, reports all samples, median and MAD.
The 93-byte corpus is a tiny regression fixture. Speed results apply to
these tested fixed shapes and this host, not modern large-model throughput.
No AVX512 or AMX speedup is claimed without hardware execution.
