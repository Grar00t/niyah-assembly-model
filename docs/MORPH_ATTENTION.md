# Actual native token-model integration

Build and verify:

```
make -f Makefile -f Makefile.morph test-morph-attention
build/morph_native/morph-train
build/morph_native/morph-infer
```

`tools/morph_attention.py` pins the original byte-model assembly by SHA-256 and
constructs a distinct u16-indexed variant. Every source adaptation requires an
exact expected match count. The build fails if the pinned source or vocabulary
changes. The original `src/attention_lm.s` is not edited or read during this
variant's build, so other model optimizations do not silently change its ABI.

The model has 294 token IDs, 16-wide embeddings, an eight-token context,
independent trainable Q/K/V matrices and the same single-head attention/FFN
architecture as the byte baseline. All forward, backward, SGD and sampling
operations execute in Assembly. Generated IDs decode to their exact surface
bytes through a native assembly lookup table. No Python process is needed to
run either produced executable. The build/test/checkpoint tools use Python's
standard library and GNU toolchain.

Checkpoint format is `niyah-morph-attention-f32-v1`: 43,480 float32 bytes,
distinct from the old 38,464-byte byte-model layout. The manifest binds model
shape, context length, vocabulary SHA-256, source, train/validation token files
and weights. Inference compiles only a separately copied and verified private
snapshot. NaN, wrong size, old schema, changed vocabulary and changed member
bytes are rejected by the packaging verifier. These hashes ensure byte
identity with the local manifest, not authenticity or general model quality.

The demonstration uses newly authored tiny related Arabic phrases. Validation
has no unseen token IDs and is deliberately not a document-independent Arabic
language benchmark. Reported losses are nats per surface token; they cannot be
compared directly with the byte model's nats per byte. Compression is measured
only against the same fixture's raw byte count, not BPE. Model context remains
eight tokens; no KV cache reduction has been measured. Training is ordinary
SGD from scratch, not continuous LoRA and not personality or factual memory.

`tests/test_morph_attention.py` checks native train/export/frozen-inference
replay, deterministic reruns, u16 IDs above255, finite parameters, distinct
learned Q/K/V changes, independent CPython forward/loss calculations, static
ELF linkage, and incompatible checkpoint rejection. Native inference freezes
learned parameters and embeds its verified model data in the executable.

## Generation encoding

Generated surface bytes accumulate in a bounded native buffer. The full buffer
is validated as strict UTF-8 before the sample marker or sample bytes are
written. Invalid sequences cause process exit2 without emitting the sample.
ASCII training metrics may already have been written. Byte fallback therefore
remains lossless for input without pretending that arbitrary sampled bytes are
a valid Unicode string. A native test-only driver checks the identical validator
against independent strict-decoding vectors, including truncation and invalid
scalar ranges.
