# Native morphology / tensor IR bridge

`niyahc-tile` is an optional fixed-shape tensor compiler in niyah-lang. It links
that checkout's real lexer and morphology functions. The existing scalar
`niyahc` grammar, parser, code generator and semantics remain unchanged.
All newly introduced executable logic is Assembly. The shared existing frontend
remains freestanding C; the additional C file contains ABI static assertions only.

Build and exercise on this checkout:

    make -C /home/a/niyah-lang-integrated-20261027 tensor-backend
    make -C /home/a/niyah-assembly-model test-ir-bridge
    /home/a/niyah-assembly-model/build/ir_bridge/demo.elf features

The four stages are: niyahc-tile --ir SOURCE -> canonical IR;
niyahc-tile --intervals IR -> NTILE001; unchanged tile_allocator -> TMM1;
niyahc-tile --asm IR ALLOCATION -> Assembly, then GNU as/ld -> static ELF64.
GNU tools and Python test oracles are not dependencies of either the new compiler
or its generated ELF at runtime. Linux x86-64 raw syscall ABI is still required;
this is not bootable bare metal.

## Grammar and identity

Optional first statement `reserve INTEGER;` sets mask0..255. Statements
`input_a NAME;`, `input_b NAME;`, `gemm NEW = A * B;`, `emit C;` form a
straight-line SSA-like tensor program. A is signed INT8[16,64], B is signed
INT8[64,16] externally row-major, and C is INT32[16,16]. Inputs are supplied as
1024-byte blocks in declaration order; emits output1024-byte INT32 blocks.
At most64 values,128 operations,128 bytes/name and1MiB source are accepted.
No forward references, name redefinition, type substitution or partial stdout.
No root, hamza or NFKC normalization aliases distinct identifiers. All available
root alternatives and the original source/span metadata are retained as
provenance, NOT selected semantic truth or numeric embeddings.

## Liveness, packing and proof

I(v)=[def(v),1+max({def(v)} union uses(v))). Operands and destination overlap
at their GEMM instruction. Values are passed to the pinned allocator in definition
order. For sorted intervals the allocator succeeds iff maximum live pressure
is at most8-popcount(mask), because an unavailable color corresponds to a live
interval at the new start; all such intervals form a clique. Allocation costs
O(8V); independent coloring validation in the bridge is O(V^2), V<=64.

The bridge verifies output count, reserved colors, range and every conflicting
pair. Actual returned TMM IDs appear in emitted TILELOADD/TILEZERO/TDPBSSD/
TILESTORED instructions. The scalar/vector schedule also uses these colors for
an eight-slot memory pool. Output snapshots are retained before slot reuse.
All tile roles share palette shape16 rows by64 bytes. B packing obeys
P[g,4*j+l]=B[4*g+l,j]; C[i,j]=sum(g,l) A[i,4*g+l]*P[g,4*j+l].
Any signed-INT8 product has absolute value at most16384; any64-term partial sum
is bounded by1048576, so INT32 accumulation is exact. Scalar, AVX2 and AVX512
use the SAME integer contract; there is no FP32 substitution or saturating i16
intermediate. LDTILECFG appears once before live tiles, TILERELEASE after them.

## IR wire contract

Little-endian NIRG0001:8-byte magic, six u32 fields for value_count, op_count,
input_count, emit_count, reserved_mask, source_bytes. Follow1024-byte definition
records,16-byte op records, and exact source bytes. Each definition contains
8 u32 (kind,def,end,name_length,line,col,candidate_count,root_count),128 bytes
for exact UTF8 name,64 triples of Unicode scalar root candidates,96 zero bytes.
Unused fields are zero. Ops are (kind,dst,a,b); kinds1/2=input,3=gemm,4=emit.
IR size =32+1024V+16O+source_bytes. Interval bytes=16+8V; assignment bytes=8+V.

Loading IR reconstructs its entire canonical representation from retained source
using the actual frontend, then compares every byte. This validates format,
metadata and liveness, but is not cryptographic authentication of supplied code.
File inputs are read-only with symlink refusal. All malformed source/IR/coloring
returns65 without stdout; usage64, input I/O66, output I/O74. The unchanged
allocator returns77 for pressure. Generated binaries return77 for explicitly
unsupported ISA,65 for input length/read error,74 on write failure; SIGPIPE is
ignored so a closed output pipe does not terminate with an unrelated signal.

## Dispatch and evidence boundary

CPUID maximum leaf and OSXSAVE are checked before leaf7/XGETBV. AVX2 requires
AVX,OSXSAVE,XCR0 bits1/2,AVX2. AVX512F requires mask0xe6 plus AVX512F.
AMX requires TILE+INT8,XCR0 tile bits,kernel-supported state, permission request
for component18 and a permission-mask recheck in the final process. Auto priority:
AMX -> AVX512 -> AVX2 -> scalar. Denied AMX permission falls back without a tile
instruction. Explicit modes never bypass these checks. `features` emits eight
u64: NXISA001,caps,XCR0,CPUID1.ECX,CPUID7.EBX,CPUID7.EDX,selected,AMX_permission.
Backend IDs0,1,2,3 correspond to scalar,AVX2,AVX512,AMX.

On i5-10400F only scalar/AVX2 run. Native B packing and576 CPUID/state combinations
are tested independently; this is NOT AMX/AVX512 hardware verification. Tests
also compare real morphology root alternatives, scalar/auto/AVX2 integer GEMM,
canonical IR corruption, conflicting colorings, all256 reservation masks,
SSA reuse, limits, I/O failures and byte-identical rebuilds.

No CFG joins/spills/dynamic shapes, trained embeddings or language-model
integration are implied. No new Arabic-quality, KV-cache or continual-learning
claim is made. This extension compiles explicit tensor programs, not arbitrary
natural-language input into matrices. Source originals, prior dirty work, old
checkpoints, Git index, branch and HEAD are preserved.

Architecture references:
https://docs.kernel.org/arch/x86/xstate.html
https://www.intel.com/content/www/us/en/developer/articles/technical/tuning-guide-for-ai-on-the-4th-generation.html
