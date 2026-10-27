# Verification record

## Recorded local evidence: 27 October 2026

These are measured implementation checks on an AVX2 Linux/WSL host, not a
claim that every input or processor has been verified.

| Gate | Recorded result |
| --- | --- |
| Standalone interval allocator | 1,915 expected outcomes |
| Core AST vs reference sidecar | 226 IR, 226 assembly and 226 whole-ELF comparisons |
| Native arithmetic vs C reference | 99 comparisons; 274,176 INT32 elements |
| Core tensor API | 26 ASan/UBSan cases |
| Unsupported tensor ISA requests | 22 correct rejections |
| Legacy scalar ELF parity | 64 comparisons when the preserved baseline was supplied |

The legacy baseline is not shipped as a binary. Its optional comparison is
not silently counted as a new fresh-clone pass. AMX and AVX-512 arithmetic
remain untested on supporting hardware. Sanitizers instrument the test C
paths, not the emitted machine code.

## Identity boundary

```text
tile_allocator
59447d2ededb37fb66985be7cbb2bb3c4e7b8326443141b61880ece2158e0866

niyahc-tile
b6924af76ea75dd3f23ef80ff926d67f74959185c346ec178c8df03ad543afe8

reference tensor ELF
460d3d202fd42d389ff03e7c6305c6c7296480752ff81bead47550187db3c0f4
```

These identities refer to recorded byte sequences and toolchains, not
universal reproducible-build guarantees or cryptographic signatures.

## Publication checks

The first source-only fresh-clone build exposed a missing allocator build
prerequisite in Makefile.ast. The compiler review branch makes that dependency
explicit. Results from the subsequent review build are recorded separately
in `publication-verification.json` after commands complete.

No credentials, pretrained weights or external datasets are intended for
publication. Filename and common-secret-pattern checks are limited checks,
not an exhaustive security audit. Machine-specific paths in the initial
public evidence snapshot were redacted; local original receipts were not rewritten.
