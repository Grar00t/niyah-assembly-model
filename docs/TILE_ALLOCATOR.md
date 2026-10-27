# Native AMX tile-register interval allocator

## Contract

This is an original x86-64 Assembly, syscall-only Linux executable. It allocates
register identifiers; it never executes AMX, AVX, or floating-point instructions.
GNU `as`/`ld` are build tools. Python's standard library is used only for tests.
The program is not a bootable bare-metal image or a general-purpose compiler.
It is not connected to niyah-lang IR, morphology, or neural inference.

Input is a finite stdin stream of exactly 16+8*N bytes:

| Offset | Width | Meaning |
| --- | --- | --- |
| 0 | 8 | ASCII `NTILE001` |
| 8 | 1 | Number of virtual values, 1..64 |
| 9 | 1 | Physical tile reservation mask; bit k reserves TMMk |
| 10 | 6 | Zero bytes |
| 16 | 8*N | Unsigned little-endian (start,end) instruction-position pairs |

Intervals are nonempty, half-open [start,end), sorted by nondecreasing start.
Equal starts are legal. At an instruction that reads an input and writes a
non-aliasing output, both live intervals must overlap that instruction. The
caller must provide conservative, correct liveness; this allocator does not
infer it. Every resource must have compatible fixed TILECFG geometry throughout
its allocation scope. No tile reconfiguration, spills, CFG analysis, alias
inference, precolored virtual values, or arbitrary interference graphs exist.

Output is exactly 8+N bytes: `TMM1`, N, three zero bytes, and N tile IDs 0..7.
The canonical three-live-values fixture maps C,A,B to TMM0,TMM1,TMM2.
The full input is validated before allocation. Rejections emit no output.
The input must reach EOF; this is not an interactive protocol.

Exit codes: 0 success; 64 extra CLI argument; 65 invalid input or input read
failure; 74 output failure (including closed pipe and inability to configure
SIGPIPE disposition); 77 valid input requiring more available registers.
Output failure cannot in general retract bytes already written to a file.
The program changes SIGPIPE disposition only in its own process.

## Deterministic algorithm and proof

For each interval in start order, choose the lowest unreserved tile whose prior
live end is <= the new start, and replace that tile's end. Check at most eight
resources per interval. Validation and allocation each take O(N) time, since
eight is fixed. N <= 64 bounds the native loop counts. No heap allocation.

Let G have an edge (i,j) exactly when max(s_i,s_j) < min(e_i,e_j).
Let R = 8 - popcount(reserved_mask) and

    omega = max_t sum_i 1[s_i <= t < e_i].

All simultaneously live intervals must use distinct colors, so omega <= R is
necessary. When the algorithm rejects a new interval starting at s, every one
of the R resources holds an interval ending after s. Because starts were sorted,
those R earlier intervals and the new interval are all live at s. Hence
omega >= R+1, so rejection is correct. Otherwise each reused color has expired
and introduces no interference. This proves success iff omega <= R and optimal
color count on this restricted interval-graph domain, not on arbitrary graphs.
The algorithm is deterministic and has no probabilistic logic.

## Corrected defects

The original candidate allocated before finishing validation. Valid high-pressure
prefixes could mask malformed later intervals with exit 77 rather than 65.
A complete validation pass now precedes coloring.

The original candidate died from SIGPIPE when its stdout pipe had no reader.
Native rt_sigaction ignores SIGPIPE, allowing the write error to return exit 74.
Both defects were reproduced on the user's Kali WSL host before correction.

## Verification

`make tile-allocator test-tile-allocator` builds and verifies the component.
The test harness checks 1,915 cases, including all 256 reservation masks, 600
seeded random schedules, a bounded exhaustive domain, unsigned boundaries,
all truncations of the eight-interval payload, malformed tails after pressure,
deterministic output, fragmented input, /dev/full, and a closed output pipe.
An independent overlap oracle checks necessity, proper coloring and minimal
color count. Two fresh assembler/linker runs must produce identical binaries.
An ELF parser verifies absence of an interpreter, dynamic segment and undefined
symbols; program headers are checked for writable+executable mappings.

The native critical coloring block is delimited by nx_color_begin/nx_color_end.
SHA-256 of that exact byte range and all .text bytes is recorded after linking.
A pre-build expected hash may be taken from an explicitly identified reference
build, but cannot honestly be invented before the bytes/layout are determined.

## AMX enablement is separate

Linux arch_prctl(ARCH_REQ_XCOMP_PERM,18) requests permission for tile data.
It does not load the 64-byte TILECFG payload: LDTILECFG does that. CPUID/XGETBV,
OS permission, data packing, valid geometry and safe dispatch remain mandatory
in a separate AMX execution kernel. This allocator running successfully on AVX2
hardware is not AMX execution evidence.

Primary references:
https://docs.kernel.org/arch/x86/xstate.html
https://www.intel.com/content/www/us/en/developer/articles/code-sample/advanced-matrix-extensions-intrinsics-functions.html
