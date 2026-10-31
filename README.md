# NIYAH Assembly Language Model

A small, **genuinely trainable next-byte neural language-model experiment**
implemented entirely in GNU x86-64 Assembly for Linux (including Ubuntu/WSL).
Weights are initialized and learned from scratch by explicit Assembly
backpropagation/SGD. No external model, libc, ML runtime, hosted API,
or MSVC `cl.exe` is involved in training or inference.

reasoning.** The current text fixture has 93 training bytes and 68 related
validation bytes. The model cannot be claimed to understand language based
on these results. Python is *only* a test/packaging/benchmark harness.

## Native Nexus integration milestone (2026-10-27)

The new optional components are executable on Linux/WSL, use raw kernel
syscalls, and do not link libc or depend on PyTorch/CUDA at runtime:

```sh
make nexus-native
make verify-nexus
./build/morph_native/morph-infer
./build/nexus_audit add 1 10 2 10 3/10
```

- `attention_lm_tiled`: four-row AVX2 Q/K/V projection, preserving the
  original path's trained weights exactly in the 240-epoch regression.
- `tile_probe`: fixed-shape scalar/AVX2 GEMM plus guarded AVX512F and
  AMX INT8 paths. The current i5-10400F executes AVX2; requests for the
  unsupported paths return 77. This is an OS-hosted static ELF, not a
  bootable bare-metal environment.
- `morph-tokenizer`: lossless Arabic surface segmentation using the
  existing user-owned niyah-lang morphology implementation, with byte
  fallback and a versioned 294-token vocabulary.
- `morph_native/morph-infer`: a distinct u16-token attention model and
  versioned checkpoint, with whole-sample strict UTF-8 output checking.
  The existing byte checkpoints are incompatible with this new model.
  Native implementation checks pass. The tiny Arabic experiment OVERFITS:
  heldout loss rises from 2.824 to 5.358 nats/token. It is not a language
  quality success or a measured KV-cache reduction.
- `continual_lora`: isolated rank-2 online projection adaptation,
 64 trainable parameters, frozen native base, bounded steps, replay
  checks, persisted generations, and rejection without overwriting the
  previous adapter. It is not yet wired into language-model generation.
- `nexus_audit`: compares a candidate rational answer with the sibling
  native exactcalc using fork/pipe/execve/wait4. Only its declared
  arithmetic domain is verified.

See [native tiles](docs/NATIVE_TILES.md),
[morphology](docs/MORPH_TOKENIZER.md),
[continual adaptation](docs/CONTINUAL_LORA.md), and
[symbolic auditor](docs/NATIVE_AUDITOR.md). Build/test orchestration uses
GNU tools and Python's standard library; the produced native executables
run independently of Python. The separate niyah-lang compiler remains
hosted C, while its generated ELF programs are syscall-only.

Measured local speed ratios and all raw samples are in
`evidence/nexus-native-benchmark.json`. They describe fixed tiny workloads,
not general large-language-model throughput. Visual context compression
is not implemented: the current text model has no trained vision encoder.

## Exact mathematical engine and checkpoint safety (October 2026)

Niyah now includes an independently testable **pure Assembly exact rational
calculator**, implemented alongside (not inside) the educational byte-level
neural model. It computes reduced fractions using checked signed64
intermediates and Euclid's GCD. Unlike floating-point, fractional answers
within its declared domain are exact:

~~~sh
make exactcalc
./build/exactcalc add 1 10 2 10    # 3/10 (exact)
./build/exactcalc add 1 3 1 6      # 1/2
./build/exactcalc div 5 7 -2 9     # -45/14
make test-exactcalc
~~~

It takes signed32 decimal numerators and positive <=2147483647
denominators; unsupported ranges, division by zero and malformed input
are deliberately rejected. **5,897 reference-oracle calculations and 24
failure vectors passed locally**. This is not an arbitrary-precision or
universal error-free mathematical system. See
[docs/EXACT_MATH.md](docs/EXACT_MATH.md).

The saved-attention inference path also closes a verification-to-use
(TOCTOU) gap by assembling from **a separately verified private snapshot**
of source, corpora and weights, not the mutable project files. Both
adversarial file-replacement tests passed locally using disposable copies.
See [docs/SECURITY_AUDIT.md](docs/SECURITY_AUDIT.md) and
`make test-attention-snapshot`. No flight hardware is operated.

## Honest evaluation on real held-out stories

The pure-Assembly attention model was evaluated on two **distinct original
TinyStories dataset splits**, using 12 training stories (8,575 bytes)
and 6 validation stories (4,212 bytes), same ASCII-lowercase preprocessing.
The network/data fetch occurs only for on-device measurement; dataset
bytes are **not committed**.

At 8 and 32 epochs, heldout losses were **2.317** and **2.315
nats/byte**, compared with a smoothed two-byte-context trigram baseline
of **2.068353** on the identical validation positions. **The neural
model loses to this statistical baseline** on these data; more epochs
did not fix the gap.

Run `make test-tinystories-eval` to verify the offline test harness.
Run `python3 tools/evaluate_tinystories.py --epochs 8` on a machine
with network access to reproduce the real dataset evaluation.
Method, fixed row IDs, masks, no-leakage caveats, license and
limitations: [docs/TINYSTORIES_EVAL.md](docs/TINYSTORIES_EVAL.md).

## Fixed-weight hybrid: external real-data check (2026-10-16)

A separate, opt-in native Assembly path now streams the actual true-byte
probabilities after training without changing normal inference. A Python
benchmark combines these with a train-only smoothed trigram using the
published count/neural language-model hybrid idea. Selection used prior
validation rows 0..5 only; on **new** TinyStories validation rows 100..107,
native-only NLL was 2.242483, trigram was 2.035952, and the 0.4/0.6
hybrid was **1.975534 nats/byte** (2.97% lower than trigram).
This is an **in-vocabulary, eight-story experiment**, NOT native hybrid
generation: the mixture itself is currently a Python evaluation step.

A second predeclared slice (rows 200..207) had unseen digits/hyphens
and could not be scored fairly under the existing masked vocabulary.
The failure was preserved, not substituted with a more convenient test.

Use `make test-hybrid-eval` for network-free native output regression,
and `python3 tools/hybrid_eval.py --epochs 32` to reproduce with pinned
Hugging Face data. Full metrics, limitations and next test:
[docs/HYBRID_EVAL.md](docs/HYBRID_EVAL.md).

## What runs on the machine?

| | Legacy trigram baseline | Native neural attention |
| --- | --- | --- |
| Model | next-byte table conditioned on 2 preceding bytes | learned one-head causal attention with 3 independently trainable Q/K/V matrices and residual FFN |
| Context | 2 bytes | 8 bytes |
| Trainable float32 weights | 16,777,216 (64 MiB) | **9,616 (38,464 bytes)** |
| Training | handwritten Assembly SGD | handwritten Assembly forward, backward and SGD |
| Generation | stochastic next byte | stochastic next byte from learned softmax |
| Linux binary | static ELF, Linux syscalls | static ELF, Linux syscalls |
| Dependencies during model run | none (kernel only) | none (kernel only) |
| Main limits | no attention | one head, one block, no real tokenizer or long context |

"GPT hides behind corporate hype and false claims of mathematical breakthroughs, but it’s nothing more than a plagiarism engine built on stolen scientific research. While they put a marketing bat in their README to trend on social media, this project actually implements independent Q/K/V matrix training for real architectural comparison. We don't need to fake miracles when the math speaks for itself."

## Build, train, test and generate — no `cl.exe`

Requirements: Linux x86-64 (or Ubuntu under Windows Subsystem for Linux),
GNU assembler+linker accessed via `cc`/`gcc`, `make`. Python 3 is only needed
for external differential tests/checkpoint packaging.

```bash
make                        # build legacy, neural attention and exact rational binaries
make exactcalc               # independent pure Assembly rational arithmetic
make test-exactcalc          # independent Fraction oracle and parser/ELF checks
make test-attention-snapshot # mutate disposable checkpoint files around compile
make run-attention-auto     # train from scratch and sample; CPUID-guarded AVX2
make run-attention          # explicitly portable scalar variant
make test-attention        # independent mathematical forward oracle
make test-qkv-gradients    # independent ONE_STEP Q/K/V derivative checks
make test-simd             # safe fallback, SIMD and numeric comparisons
make attention-checkpoint  # run native Assembly training, persist weights
make verify-attention      # SHA-256, data and model contract checks
make test-attention-checkpoint test-attention-negative
make generate-attention-auto  # inference with saved weights, no retraining
make verify-all            # full regression gate for both models
```

The saved raw-float32 weight file is `models/attention_weights.f32`, which
is intentionally not committed. The manifest version is
`niyah-attention-f32-v2-learned-qkv` and requires **38,464 bytes**; 35,392-byte
v1 files are intentionally rejected rather than silently reinterpreted.
Standalone inference embeds verified weights with GNU Assembly `.incbin`.

For legacy trigram functionality use `make run test`, then
`make checkpoint verify test-checkpoint generate`.

## Measured on the user's machine (2026-10-15)

Machine: Intel Core i5-10400F, Ubuntu under WSL. One fixed deterministic
initialization, **240 SGD epochs** on the 93-byte toy training set.

| Metric (nats per byte) | Before | After |
| --- | ---: | ---: |
| Training cross entropy | 2.500 | **0.167** |
| Related heldout cross entropy | 2.498 | **0.238** |

An independent CPython forward pass over the *Assembly-trained parameters*
recomputed scalar train `0.166523` and heldout `0.238447` nats/byte.
The tested three independently trained Q/K/V gradient coordinates matched
central finite differences with relative error approximately `10^-7`.
All 4 checkpoint-corruption negative tests rejected modified/invalid data.
These results are about a tiny **related** text fixture, not a meaningful
natural-language generalization benchmark or a claim that the model can
invent independent facts. Details: [docs/ATTENTION.md](docs/ATTENTION.md).

### Genuine native AVX2 acceleration with scalar fallback

The original GNU Assembly `dot16` kernel powers Q/K/V projection, attention
score, FFN and output projection. On supported CPUs, `CPUID` and `XGETBV`
confirm AVX2/FMA and OS support before taking the vector path; the same
binary falls back to scalar SSE on unsupported machines. No third-party
Assembly source was copied.

Repeated native CPU microbenchmark: **1,200 epochs**, 12 order-rotated
warm-run samples per path, measured median wall time:

| Path | Median | Compared with scalar |
| --- | ---: | ---: |
| Scalar | 0.999423 s | 1.00x |
| CPUID-dispatched AVX2/FMA | **0.792365 s** | **1.2613x faster** |
| Explicit AVX2/FMA | 0.781844 s | 1.2783x faster |

All 36 durations and environment information are in
[evidence/simd-benchmark-local.json](evidence/simd-benchmark-local.json)
and the implementation/limitations in [docs/SIMD.md](docs/SIMD.md).
The 1.2613x result is for *this* small M3 workload and hardware, not
general LLM throughput.

## Limitations and evidence boundary

- Eight-byte context; input/output is **raw bytes**. Output vocabulary
  is masked to bytes seen in the small training fixture. This is not a BPE
  tokenizer, semantic text understanding, or UTF-8 guaranteed decoding.
- One attention head, one attention operation, 16-wide hidden state and
  residual ReLU FFN. No stacked decoder blocks, LayerNorm/RMSNorm,
  optimizer state, or multi-head attention.
- Holdout is 68 very similar bytes and is *not* independent document-level
  evaluation. Quality beyond its familiar tiny phrases has not been shown.
- The older M2 feature (fixed identity Q/K/V, 8,848 weights) is preserved
  in Git history [PR #3](https://github.com/Grar00t/niyah-assembly-model/pull/3);
  **v2** represents a deliberately changed weight layout.
- Checkpoint SHA-256 is an integrity measure relative to the stored manifest,
  not a model-quality certificate, signature or general memory-safety proof.
- Running Python test scripts does **not** make the assembly executable
  a Python model; all native forward/backprop/SGD kernels execute in Assembly.

## Sources and next experiment

Assembly optimizations learned from the approach in MIT-licensed
[NN_S](https://github.com/zenyyxz/NN_S), the worklog at
[matmul_assembly_x86](https://github.com/alint77/matmul_assembly_x86) and
[Intel's Optimization Reference Manual](https://www.intel.com/content/www/us/en/content-details/671488/intel-64-and-ia-32-architectures-optimization-reference-manual-volume-1.html).
**No source copied**. Further compact-transformer benchmarks could use the
[HF TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
(subject to its `cdla-sharing-1.0` license and documented train/test separation),
but it has **not** been used for the measurements above, nor have any GPUs
been rented. A serious multi-block language model will require a separate
design, larger licensed corpus and repeatable heldout eval, not a new label
on this byte predictor.

The near-term next gate is a second genuine causal decoder block plus
numerical gradient validation, while preserving this entire baseline.
