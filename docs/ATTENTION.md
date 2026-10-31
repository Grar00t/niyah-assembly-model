# Byte attention model

`src/attention_lm.s` implements embeddings, learned Q/K/V projections, causal attention, a residual ReLU feed-forward layer, masked output probabilities and SGD. The executable uses Linux syscalls. Python builds it, packages checkpoints and supplies independent test oracles.

## Current source and regression

The current source sets `CTX=32`; `tests/test_attention.py` sets `T=8`, and the checkpoint helper still declares an eight-byte context. `POS` reserves 512 bytes and `TOK` reserves 8 bytes; those reservations do not cover the declared context. The fixed inference offsets and parameter-size checks also retain the prior layout. This is a source/format mismatch, not a validated context extension. [`evidence/docs-refresh-20261031/contract-excerpts.json`](../evidence/docs-refresh-20261031/contract-excerpts.json)

The current corpus contains 6652 training bytes and 740 heldout bytes. These differ from the historical evaluation fixtures. [`evidence/docs-refresh-20261031/inspection.json`](../evidence/docs-refresh-20261031/inspection.json)

`make verify-all` currently fails. Read the failing assertion and command in [`evidence/docs-refresh-20261031/verify-all.log`](../evidence/docs-refresh-20261031/verify-all.log). The calculator repair does not change the byte-model implementation, oracle or checkpoint formats.

## Historical measurements

These receipts describe prior source/data snapshots. They are not results from the current source.

| Quantity | Recorded value | Evidence |
| --- | --- | --- |
| Learned-QKV train loss | 2.500 before; 0.167 after, nats/byte | [`evidence/attention-learned-qkv-local-20261015.json`](../evidence/attention-learned-qkv-local-20261015.json) |
| Learned-QKV heldout loss | 2.498 before; 0.238 after, nats/byte | [`evidence/attention-learned-qkv-local-20261015.json`](../evidence/attention-learned-qkv-local-20261015.json) |
| Independent train / heldout loss | 0.166523 / 0.238447 nats/byte | [`evidence/attention-learned-qkv-local-20261015.json`](../evidence/attention-learned-qkv-local-20261015.json) |
| Context intervention | Distribution L1 difference 1.10443155 | [`evidence/attention-learned-qkv-local-20261015.json`](../evidence/attention-learned-qkv-local-20261015.json) |
| WQ / WK gradient relative error | 2.1365569640984e-7 each | [`evidence/qkv-gradients-local-20261015.txt`](../evidence/qkv-gradients-local-20261015.txt) |
| WV gradient relative error | 5.6357535968417204e-8 | [`evidence/qkv-gradients-local-20261015.txt`](../evidence/qkv-gradients-local-20261015.txt) |
| One-step loss | 2.5178064973076397 to 2.491438666899255 | [`evidence/qkv-gradients-local-20261015.txt`](../evidence/qkv-gradients-local-20261015.txt) |
| TinyStories architecture | 9616 parameters; 8-byte context | [`evidence/tinystories-external-32epochs-local.json`](../evidence/tinystories-external-32epochs-local.json) |

The complete older attention receipt also remains in [`evidence/attention-local-20261015.json`](../evidence/attention-local-20261015.json). [TinyStories](TINYSTORIES_EVAL.md) records the neural model losing to the smoothed trigram. [SIMD](SIMD.md) records the historical timing samples.

## Checkpoints and commands

The historical learned-QKV format requires 38464 bytes; the pinned morphology variant uses a separate format. The current source must be reconciled with its checkpoint/oracle layout before treating a newly written byte checkpoint as compatible. [`evidence/docs-refresh-20261031/contract-excerpts.json`](../evidence/docs-refresh-20261031/contract-excerpts.json)

```sh
make test-attention test-qkv-gradients test-simd
make test-attention-snapshot
```

These are diagnostic targets and may fail on the current source. `attention-checkpoint` writes under the checkout's `models/`; preserve existing files before any training run. SHA-256 checks bind bytes to a local manifest. They do not authenticate the manifest. [Checkpoint review](SECURITY_AUDIT.md).
