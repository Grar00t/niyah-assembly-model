# Niyah | Native kernels and model experiments

Linux x86-64 Assembly for fixed-shape tensor execution, small trainable
attention models, guarded adaptation and exact rational arithmetic.
Native numerical engines use raw Linux syscalls, not libc, CUDA or PyTorch.
The surface tokenizer combines freestanding C with an Assembly entry point.

[Kernels](docs/NATIVE_TILES.md) / [Tensor IR](docs/TENSOR_IR.md) /
[Allocation](docs/TILE_ALLOCATOR.md) / [Evidence](docs/VERIFICATION.md)

```text
niyah-lang                              niyah-assembly-model
Arabic source -> AST -> typed IR  ---->  interval allocation
                                        AMX / AVX-512 / AVX2 / Scalar
                                        static Linux ELF64
```

## Build

The review branches form a matched source pair. Neither has been merged to main.

```sh
git clone --branch review/nexus-native-20261027 https://github.com/Grar00t/niyah-assembly-model.git
git clone --branch review/nexus-ast-core-20261027 https://github.com/Grar00t/niyah-lang.git
make -C niyah-assembly-model nexus-native NIYAH_LANG=../niyah-lang
make -C niyah-assembly-model verify-nexus NIYAH_LANG=../niyah-lang
```

Requirements: Linux x86-64 or WSL, GNU assembler/linker, GCC-compatible `cc`,
GNU Make, and Python 3.10+ standard library for build/test orchestration.
No Python interpreter is used by the resulting native model executables.
See [integration-lock.json](docs/integration-lock.json) for compatible source pins.

## Components

| Component | Contract | Boundary |
| --- | --- | --- |
| `tile_allocator` | Sorted half-open intervals; 8 resources; explicit reserved mask | No spills or control-flow joins |
| Tensor runtime/dispatch | Signed INT8 16x64 x 64x16 -> INT32 16x16 | AMX/AVX-512 arithmetic not hardware-validated in the audit |
| `attention_lm_tiled` | Four-row AVX2 Q/K/V projection | Tiny-model results are not LLM throughput claims |
| `morph-tokenizer` | Reversible Arabic surface segmentation; 294-token vocabulary | Candidate roots are not embeddings |
| `morph_native/morph-infer` | Separate u16-token attention experiment | Heldout quality gate failed through overfitting |
| `continual_lora` | Rank-2 projection adaptation with retention/rejection gates | Not connected to language generation |
| `nexus_audit` / `exactcalc` | Bounded checked rational arithmetic | Not a general factual verifier |

## Run a bounded check

```sh
./build/exactcalc add 1 10 2 10
# 3/10
./build/nexus_audit add 1 10 2 10 3/10
# VERIFIED 3/10
make test-tile-allocator test-ir-bridge NIYAH_LANG=../niyah-lang
```

The default native tensor dispatcher prefers AMX, then AVX-512, AVX2 and
Scalar only when CPU/OS state permits. Explicit unsupported modes return 77.
These programs require Linux; static ELF does not mean bootable bare metal.

## Evidence, not labels

The Arabic fixture's heldout loss rose from 2.824 to 5.358 nats/token.
Implementation checks passed; language generalization did not. Byte reduction
on the small fixture is not a measured KV-cache reduction or BPE comparison.

Earlier native attention, TinyStories and hybrid evaluations are retained in
[Historical experiment notes](docs/HISTORY_NATIVE_MODEL.md). Their distinct
corpora and metrics must not be combined into one performance claim.
Current gates and evidence files are indexed in [Verification](docs/VERIFICATION.md).

Weights, generated executables and external dataset payloads are not published.
Tests may create small native fixture weights locally. Real-data evaluation
is an explicit opt-in path; normal regression gates do not fetch datasets.

## Review and provenance

The companion compiler now incorporates tensor intrinsics in `niyahc` rather
than launching a sidecar. This repository retains the reference backend and
its differential oracles. [Morphology provenance](src/morph_tokenizer/upstream/PROVENANCE.json)
and its MIT license are preserved. First-party copied sources must not be
described as having no provenance.

GitHub Actions is manual-only on this branch. No automatic merge is enabled.

## License

MIT. See LICENSE and the retained notices beside included sources.
