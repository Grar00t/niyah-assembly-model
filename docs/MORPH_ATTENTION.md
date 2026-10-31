# Morphology token-model experiment

`tools/morph_attention.py` adapts the pinned `src/morph_tokenizer/upstream/attention_byte_v2.s` through checked substitutions. It does not use the current `src/attention_lm.s` as its source. Training, backpropagation and sampling execute in the resulting Assembly binary; Python builds and verifies it.

```sh
make morph-attention test-morph-attention
build/morph_native/morph-infer
```

The checkpoint contains 43480 bytes and uses 294 token IDs. The fixture contains 198 training bytes / 61 tokens and 74 validation bytes / 23 tokens. [`evidence/nexus-morph-attention.json`](../evidence/nexus-morph-attention.json)

| Measurement | Nats/token | Evidence |
| --- | ---: | --- |
| Initial train loss | 2.827 | [`evidence/nexus-morph-attention.json`](../evidence/nexus-morph-attention.json) |
| Final train loss | 0.030 | [`evidence/nexus-morph-attention.json`](../evidence/nexus-morph-attention.json) |
| Untrained heldout loss | 2.824 | [`evidence/nexus-morph-attention.json`](../evidence/nexus-morph-attention.json) |
| Trained heldout loss | 5.358 | [`evidence/nexus-morph-attention.json`](../evidence/nexus-morph-attention.json) |

Heldout loss increased. Independent oracle values, source/model hashes, replay results and incompatible-checkpoint rejections remain in the same receipt. Loss per surface token is not directly comparable with loss per byte.

Generation buffers the complete sample and checks strict UTF-8 before emitting the sample marker and bytes. Invalid generated encoding is rejected; earlier diagnostic metrics may already be written. Byte-preserving input does not imply valid arbitrary sampled output.

The fixture is not a document-independent Arabic quality benchmark. No BPE advantage, KV-cache reduction or continual adaptation was measured. [Codec](MORPH_TOKENIZER.md); [sources](SOURCES.md).
