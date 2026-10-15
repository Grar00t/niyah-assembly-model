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

## New: runtime-dispatched AVX2/FMA3 Assembly kernels (2026-10-15)

Three real linear-algebra dot-products (attention Q/K, residual FFN and
output projection) now share a hand-written x86-64 `dot16` kernel.
`make run-attention-auto` uses `CPUID` and `XGETBV` to enable AVX2+FMA3
only when the CPU **and OS** support it; otherwise it runs the safe scalar
fallback. `make test-simd` compares both Assembly paths to an independent
floating-point reference and checks training/dispatch reproducibility.

On one local WSL machine, 1,200 training epochs with 12 interleaved wall-clock
repetitions took 0.395372 s (scalar median) versus 0.363059 s
(CPUID-dispatched AVX2 median): **1.089x speedup for this tiny workload**.
AVX2/SSE numerical associations
differ slightly; they are not claimed to be bit-identical, but all relevant
loss and gradient regression tests passed locally.

The fast kernel is inspired by (not copied from)
[NN_S](https://github.com/zenyyxz/NN_S) (MIT) and Intel's published
SIMD optimization guidance. Detailed CPU checks, numerical tolerances and
all raw timing samples are in [docs/SIMD.md](docs/SIMD.md) and
[evidence/simd-benchmark-local.json](evidence/simd-benchmark-local.json).
This is a **kernel optimization, not an increase in model intelligence**.

## Run the real program locally

Requires Linux x86-64 (or Ubuntu on WSL), GCC or compatible C driver with GNU
assembler and linker, GNU make, and Python 3 **only for verification/checkpoint
packaging**. Neither training executable requires a Python runtime.

```bash
make
make run-attention-auto   # recommended: AVX2/FMA3 when safe, scalar otherwise
make run-attention        # explicit portable scalar training and sample
make test-simd            # independent scalar/SIMD mathematical checks
make test-attention       # independent math-oracle verification
make attention-checkpoint
make verify-attention
make test-attention-checkpoint
make generate-attention-auto  # pure Assembly inference, runtime-dispatched
make generate-attention   # pure scalar Assembly inference from saved weights; no training
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

The independent Python **test oracle** in the original scalar release
recomputed 0.174263 and 0.240447
nats/token from exported Assembly-trained float32 weights. Small floating-
point association changes in the new shared dot16 scalar kernel yield
0.174228 and 0.240282. The normalized
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
