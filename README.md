# Niyah Assembly Model

A small **learnable neural language-model experiment**, trained and run in GNU x86-64
assembly on Linux without libc. It is **not a large language model**, a chatbot, or
a proof of novel reasoning. The repository retains the original byte-trigram
baseline and adds a neural **single-head causal-attention** LM with a residual
ReLU feed-forward layer.

## What is real, and what is not

| | Existing trigram | New attention LM |
| --- | --- | --- |
| Architecture | next-byte softmax table indexed by last two bytes | learned token/position embeddings, 8-byte causal attention, residual FFN, softmax |
| Trainable weights | 16,777,216 float32 (64 MiB) | 8,848 float32 (35,392 bytes) |
| Training | SGD entirely in assembly | forward, reverse-mode derivatives, SGD entirely in assembly |
| Context | 2 input bytes | 8 preceding bytes |
| Runtime | freestanding x86-64 ELF, direct Linux syscalls | freestanding x86-64 ELF, direct Linux syscalls |
| Heldout evaluation | not included in baseline | related but separate heldout text (NOT a generalization benchmark) |
| Model class | n-gram predictor, not neural context mixing | small attention-based neural next-byte predictor (NOT a full GPT) |

**Important architectural limit:** the new attention model uses **identity-tied
Q/K/V projections** (it does not train full Wq/Wk/Wv), one head, one attention
operation, and a 16-wide residual ReLU feed-forward layer. There are no
multi-layer decoder blocks, LayerNorm, BPE, extended context, or instruction
tuning. Generation is stochastic sampling from trained next-byte probabilities,
not a demonstrated capability to discover knowledge outside the weights/input.

## Run the real program locally

Requires Linux x86-64 (or Ubuntu on WSL), GCC or compatible C driver with GNU
assembler and linker, GNU make, and Python 3 **only for verification/checkpoint
packaging**. Neither training executable requires a Python runtime.

```bash
make
make run-attention        # train from deterministic initialized weights and sample
make test-attention       # independent math-oracle verification
make attention-checkpoint
make verify-attention
make test-attention-checkpoint
make generate-attention   # pure Assembly inference from saved .incbin weights; no training
```

To run the untouched trigram baseline:

```bash
make run
make test
make checkpoint
make verify test-checkpoint
make generate
```

Changing `data/corpus.txt` requires rebuilding the model. The small
`data/heldout.txt` is evaluated after training; it intentionally uses related
phrases and vocabulary, not a strong semantic holdout. The corpus is read as
**raw bytes** by `.incbin`, no tokenizer / UTF-8 constrained decoding.

Compiled inference embeds `models/attention_weights.f32`, whose raw output is
produced by the Assembly trainer; it is not a wrapper for Python, an API, or a
hidden pretrained provider. The checkpoint manifest hashes source, training
data, validation data and all weights. `models/` is intentionally gitignored.
A manifest and file are replaced atomically *individually*, not transactionally;
a partial/crashed update can therefore fail verification. SHA-256 is an integrity
check, not a signature or proof of capability.

## Measured local result (one machine, one seed)

With the 93-byte training text and 68-byte related heldout text, 240 epochs,
the local executable reported negative log-likelihood in **milli-nats**:

- Initial training: 2.500 nats/token.
- Final training: 0.174 nats/token.
- Heldout before training: 2.498 nats/token.
- Heldout after training: 0.240 nats/token.

The independent Python **test oracle** recomputed 0.174263 and 0.240447
nats/token from exported Assembly-trained float32 weights. The normalized
output distributions for two contexts with identical final two bytes
(`the cat ` vs `the rat `) differed in L1 by 0.83645707: this is not
a two-byte lookup table. The fixture is far too small to claim language
understanding, novelty, field usefulness, or broad transfer. See
`tests/test_attention.py` for exact tests and `docs/ATTENTION.md` for
math and limitations.

## Other reference implementations

- [sectorllm](https://github.com/rdmsr/sectorllm) is a boot-sector Assembly
  **inference** implementation for a real compact Transformer; it does not
  establish from-scratch Assembly **training**.
- [TinyStories](https://arxiv.org/abs/2305.07759) demonstrates the role of
  more substantial training and validation data for compact language models.
  Training on TinyStories is future work, not done here.
- Drone flight stacks such as [PX4](https://docs.px4.io/main/en/concept/architecture)
  use primarily C/C++ components; Assembly can optimize critical kernels,
  but there is no general proof that drones are mostly written in Assembly.

Keep model quality, training mechanism, cross-domain generalization, external
software, and costs as separately testable claims.
