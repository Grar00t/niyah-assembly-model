# Native Arabic morphology bridge (experimental vocabulary v1)

`make -f Makefile -f Makefile.morph test-morph-tokenizer` builds and tests a
static x86-64 ELF that uses Linux syscalls directly. GCC/as/ld and Python are
development tools. The executable links neither libc nor Python, PyTorch or
CUDA. It is a Linux/WSL userspace program, not a bootable bare-metal kernel.

## Exact-byte contract

```
build/morph-tokenizer encode < input.txt > input.nymt
build/morph-tokenizer decode < input.nymt > reconstructed.txt
build/morph-tokenizer vocab
```

The codec preserves all bytes, including diacritics, invalid UTF-8, NUL,
punctuation and every whitespace spelling. It accepts at most 1,048,576 input
bytes per invocation. Token IDs 0..255 mean the original single byte. IDs
256..293 denote the 38 exact surface strings in `vocab.json`. The morphological
analyzer supplies candidate boundaries. The codec selects the candidate with
the fewest tokens, breaking ties by original analyzer order. This choice is
for deterministic coding, not linguistic disambiguation. Roots are never
substituted for surface forms. Each unmatched surface span uses byte fallback.

The 64-byte little-endian header is: 8-byte `NYMORPH\0`, u32 format version,
u32 vocabulary version, u64 token count, u64 original byte length, and 32-byte
SHA-256 of the canonical vocabulary file. Payload is exactly `count` u16 IDs.
Decode validates the entire stream before writing output. The vocabulary hash
binds ID meanings; it does not authenticate payload contents.

Exits: 0 success; 2 malformed/incompatible token stream; 3 I/O failure;
4 input-size limit; 64 invalid command line. Broken pipes use the OS's default
SIGPIPE behavior. No claim of a universal exit3 for an unhandled signal.

## Provenance and limits

`upstream/morph.c`, `morph.h`, `lexer.h` and LICENSE are exact copies from the
user's niyah-lang checkout. `upstream/PROVENANCE.json` records repository HEAD,
source paths and each SHA-256. The MIT copyright notice is retained. The
original checkout is not modified.

This is a deliberately tiny fixed vocabulary, not a trained BPE tokenizer.
Fixture compression against raw bytes says nothing about comparisons to BPE,
general Arabic accuracy, or a measured model KV cache reduction. Multi-analysis
morphology includes ambiguous cases and a small seed verb lexicon.

**The existing 256-byte model/checkpoint is incompatible with these token IDs.**
The separate optional morphology attention experiment uses its own vocabulary,
embedding/output geometry, corpus artifacts and checkpoint schema. No existing
weights are silently reinterpreted.
