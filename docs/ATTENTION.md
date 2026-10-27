# Trainable single-head attention LM in pure x86-64 GNU Assembly

## Precise model contract (M3)

Native Linux x86-64 ELF training and inference do not use libc, PyTorch,

### Architecture, parameter layout and compute

- Context length `CTX=8` bytes, hidden size `D=16`, possible byte IDs 0..255.
- Trainable embeddings: `E[256,16]` and positional embeddings `POS[8,16]`.
- Independently trainable projections: `WQ[16,16]`, `WK[16,16]`, `WV[16,16]`.
  Each starts as the identity matrix (for a controlled transition from M2)
  but is updated by its own handwritten gradient in assembly during SGD.
- Output/FFN: `F[16,16]`, `FB[16]`, `O[256,16]`, `OB[256]`.
- Total = `4096+128+256+16+4096+256+256+256+256 = 9616` float32
  parameters, `38464` bytes of raw weights.
- `x_t = E[token_t] + POS[t]`
- `q = WQ @ x_last`, `k_t = WK @ x_t`, `v_t = WV @ x_t`
- `a = softmax( [q·k_t / sqrt(16)] for t=0..7 )`.
- `h = x_last + sum_t a_t v_t`.
- `y = h + ReLU(F @ h + FB)`.
- `p = masked_softmax(O @ y + OB)`. The vocabulary mask permits
  **only byte IDs present in the compiled training corpus**.
- Training objective: observed target's next-byte negative log probability,
  SGD learning rate `0.025` for 240 epochs, deterministic initialization.
- Training window consists solely of the eight preceding bytes, so there is
  no causality leak from the target position.

### Training and projection derivatives

All parameter updates and reverse-mode intermediate derivatives are in
`src/attention_lm.s`. The backward chain goes through the output softmax,
residual FFN, value projection, attention softmax, key projection, query
projection, residual and both embeddings. Input gradients are computed using
old projection weights **before** any Q/K/V weight update.

A test-only `ONE_STEP=1` assembler option executes exactly one native SGD
update; it is deliberately not the normal training mode. It lets
`tests/test_qkv_gradients.py` compare an off-diagonal coordinate in each of
`WQ`, `WK` and `WV` against an independently computed central finite
difference of the Python mathematical reference. This is **representative
gradient validation**, not an exhaustive formal proof for all tensor entries
or all possible inputs.

### Checkpoint v2 (incompatible with the earlier M2 release)

`tools/checkpoint_attention.py` stores the raw training output in
`models/attention_weights.f32` as contiguous float32 in the exact order above.
The separate JSON manifest identifies
`format=niyah-attention-f32-v2-learned-qkv`, includes source, dataset and
weights SHA-256, and requires exactly **38,464 bytes**.

Older v1 checkpoints (35,392 bytes, identity-tied Q/K/V) do not pass the
v2 size/format check. Do not silently migrate or mislabel an old checkpoint:
regenerate with `make attention-checkpoint`. Training assembly emits weights
through sys_write. The Python helper packages/verifies bytes but is not the
training engine. The inference binary embeds the verified weights using
`.incbin` and does not retrain.

Checkpoint negative tests operate on disposable copies and confirm rejection
of modified parameters, truncated parameter files, tampered manifest source
digest, and missing weights. SHA-256 verifies content integrity against the
manifest, **not** cryptographic authenticity or model quality.

### Replication

```sh
make run-attention-auto
make test-attention          # independent full forward math oracle
make test-qkv-gradients      # representative numerical derivative checks
make test-simd               # scalar versus AVX2 + independent math oracle
make attention-checkpoint verify-attention
make test-attention-checkpoint test-attention-negative
make generate-attention-auto # saved checkpoint, native inference
make verify-all
```

`make test` and `make checkpoint verify test-checkpoint` still test the legacy
64 MiB trigram model. The CPUID-dispatched dot16 routine is described in
[SIMD.md](SIMD.md). Python is test-only and not called by model execution.

### Exact scope of results (one 93-byte training fixture)

- Initial next-byte loss: `2.500` nats/byte.
- After 240 epochs, native scalar: displayed training `0.167` and heldout
  `0.238` nats/byte, independent Python numerical oracle `0.166523` and
  `0.238447` respectively.
- Starting heldout loss: `2.498` nats/byte.
- WQ/WK/WV separately leave identity; representative numerical SGD
  gradient error was below 3e-7 relative on tested coordinates.
- Context intervention holding trailing two bytes constant changed
  the learned next-byte probability distribution (L1 `1.10443155`).
- With AVX2/FMA3 the independent oracle found train/heldout
  `0.166541` / `0.238523` nats/byte; these minor changes are FP32
  non-associativity and may grow over an SGD trajectory.

**This is not a holdout-generalization benchmark.** Validation is only 68
bytes with closely related phrases and uses the same tiny vocabulary.
The results do not show semantic understanding, reasoning, instruction
following, novel facts, invention outside the data/weights, or scalability.

## Engineering gaps / next gate

No multi-head attention, stacked decoder blocks, LayerNorm/RMSNorm, learned
tokenizer, long-context training, optimizer state/rewind, or dataset-streaming
path yet. This model must not be marketed as GPT-scale or production-ready.
A future milestone could implement a second decoder block and test
document-level heldout corpora with independent full gradient checks;
any proposed rented GPU should have user approval and a concrete budget.
