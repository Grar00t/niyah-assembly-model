Assembly kernels, byte language-model experiments, and a Python/C INT8 transformer validation runtime for Linux.

# niyah-assembly-model

Python orchestrates and loads weights; C computes norm, RoPE, attention, scaling; Assembly computes INT8 GEMM.
This split describes the Qwen path. The byte-model and rational-calculator executables use Linux syscalls.

## Components

| Path | Purpose | Status | Evidence file |
| --- | --- | --- | --- |
| `tools/qwen/qwen_native_full_gate.py` | Teacher-forced transformer comparison | g32 PASS; per-token FAIL | [`tools/qwen/verify/full_gate_g32all_20261031_133521.json`](tools/qwen/verify/full_gate_g32all_20261031_133521.json); [`tools/qwen/verify/full_gate_pertoken_20261031_132909.json`](tools/qwen/verify/full_gate_pertoken_20261031_132909.json) |
| `src/niyah_i8_gemm.s` | Blocked signed integer matrix products | Serial kernel checks PASS | [`evidence/kernel-suite-20261031-112031-843492/results-o2-full/verification.log`](evidence/kernel-suite-20261031-112031-843492/results-o2-full/verification.log) |
| `src/attention_lm.s`, `src/trigram.s` | Byte-model training and sampling | Trigram checks PASS; attention regression FAIL | [`evidence/docs-refresh-20261031/verify-all.log`](evidence/docs-refresh-20261031/verify-all.log); [`evidence/tinystories-external-32epochs-local.json`](evidence/tinystories-external-32epochs-local.json) |
| `src/exactcalc.s`, `src/nexus_audit.s` | Reduced rational arithmetic and candidate comparison | Restored calculator and auditor checks PASS | [`evidence/docs-refresh-20261031/exactcalc.log`](evidence/docs-refresh-20261031/exactcalc.log) |
| `src/tile_kernels.s`, `src/tile_probe.s` | Matrix kernels and ISA dispatch | Recorded scalar/AVX2 checks PASS; AMX/AVX512 unexecuted | [`evidence/nexus-kernels.json`](evidence/nexus-kernels.json) |
| `src/morph_tokenizer/` | Byte-preserving surface codec | Recorded checks PASS | [`evidence/nexus-morph-tokenizer.json`](evidence/nexus-morph-tokenizer.json) |
| `tools/morph_attention.py` | Token-model build from a pinned byte-model source | Recorded execution PASS; heldout loss increased | [`evidence/nexus-morph-attention.json`](evidence/nexus-morph-attention.json) |
| `src/continual_lora.s` | Projection adaptation with replay bounds | Implementation present; no finished receipt in the evidence directories | [`evidence/docs-refresh-20261031/inspection.json`](evidence/docs-refresh-20261031/inspection.json) |
| `src/tile_allocator.s`, `Makefile.ir` | Register assignment and external compiler bridge | Source present; compiler checkout required | [`evidence/docs-refresh-20261031/inspection.json`](evidence/docs-refresh-20261031/inspection.json) |

KV-cache decode is in progress. The local `tools/qwen/qwen_native_decode_kv.py` is untracked and has no finished report in this branch. [`evidence/docs-refresh-20261031/inspection.json`](evidence/docs-refresh-20261031/inspection.json)

## Quick start

Use Linux/WSL, GNU `make`, `cc`, `as`, `ld`, `readelf`, `nm`, and Python for verification.
Run from the repository root. Checkpoint targets write under the checkout's `models/` directory; use a disposable checkout to preserve existing checkpoints.

```sh
make exactcalc test-exactcalc test-nexus-audit
make verify-all
```

The calculator checks pass. The current full regression fails; see [`evidence/docs-refresh-20261031/verify-all.log`](evidence/docs-refresh-20261031/verify-all.log) and [byte-model status](docs/ATTENTION.md). CI is manual dispatch; these receipts came from local execution.

The Qwen harness additionally needs `numpy`, `torch`, `safetensors`, `transformers`, GCC with OpenMP, and local model files. It reads those files without training them. It currently resolves the GEMM source from `$HOME/niyah-assembly-model/src/niyah_i8_gemm.s`; see [Qwen reproduction](docs/QWEN.md).

```sh
export NIYAH_MODEL_DIR="$HOME/models/Qwen3-8B-Base"
export NIYAH_REPORT="$(mktemp -d)/full-gate.json"
NIYAH_G32_ALL=1 python3 tools/qwen/qwen_native_full_gate.py
```

The environment switch and model identifier are recorded in [`tools/qwen/verify/full_gate_g32all_20261031_133521.json`](tools/qwen/verify/full_gate_g32all_20261031_133521.json). Reports use exclusive creation. A completed report's `result.gate` determines PASS or FAIL; the script does not convert a failed numerical gate into a failing process exit.

## Results

The Qwen comparison covers 36 decoder layers, final RMSNorm and the output projection: 10 prompts (5 Arabic, 5 English), 32 teacher-forced positions each. Reference: layer-streamed `transformers` FP32 eager execution. [`tools/qwen/verify/full_gate_g32all_20261031_133521.json`](tools/qwen/verify/full_gate_g32all_20261031_133521.json)

| Experiment | Measurement | Result | Evidence file |
| --- | --- | --- | --- |
| Qwen g32 activations at every linear | Top-1 0.959375; mean KL 0.0053047664 nats; top-5 overlap 0.958125 | PASS | [`tools/qwen/verify/full_gate_g32all_20261031_133521.json`](tools/qwen/verify/full_gate_g32all_20261031_133521.json) |
| Qwen per-token activations, g32 down projection | Top-1 0.853125; mean KL 0.6338464501 nats; top-5 overlap 0.875625 | FAIL | [`tools/qwen/verify/full_gate_pertoken_20261031_132909.json`](tools/qwen/verify/full_gate_pertoken_20261031_132909.json) |
| Qwen g32 process peak RSS | 3772.6 MiB (`peak_rss_mb`; Linux KiB divided by 1024) | Measured process, including reference | [`tools/qwen/verify/full_gate_g32all_20261031_133521.json`](tools/qwen/verify/full_gate_g32all_20261031_133521.json); [`evidence/docs-refresh-20261031/contract-excerpts.json`](evidence/docs-refresh-20261031/contract-excerpts.json) |
| Historical TinyStories byte attention / smoothed trigram | 2.315 / 2.068353 nats/byte, rounded 2.32 / 2.07 | Neural model loses | [`evidence/tinystories-external-32epochs-local.json`](evidence/tinystories-external-32epochs-local.json) |
| Historical fixed attention/trigram mixture | 1.975534497 vs trigram 2.035952147 nats/byte | Lower loss on the recorded in-vocabulary slice | [`evidence/hybrid-fresh-test-local.json`](evidence/hybrid-fresh-test-local.json) |
| Morphology token model | Untrained heldout 2.824; trained heldout 5.358 nats/token | Heldout loss increased | [`evidence/nexus-morph-attention.json`](evidence/nexus-morph-attention.json) |

[Byte-model measurements](docs/ATTENTION.md), [TinyStories protocol](docs/TINYSTORIES_EVAL.md), [hybrid failure](docs/HYBRID_EVAL.md), [kernel measurements](docs/SIMD.md).

## Written in this repository

- `src/niyah_i8_gemm.s`: blocked integer GEMM implementation.
- `tools/qwen/qwen_native_full_gate.py`, `tools/qwen/qwen_native_layer0_g32_tany.py`: native layer bindings, weight loading and numerical comparison harnesses.
- `src/attention_lm.s`, `src/trigram.s`: byte-model implementations.
- `src/exactcalc.s`, `src/nexus_audit.s`: bounded rational calculator and candidate comparison.
- `src/tile_kernels.s`, `src/tile_probe.s`, `src/tile_allocator.s`, `src/tensor_runtime.s`, `src/tensor_dispatch.s`: kernels, allocation and dispatch implementations.
- `src/morph_tokenizer/tokenizer.c`, `src/morph_tokenizer/start.s`, `tools/morph_attention.py`: codec wrapper and token-model build adaptation.
- `src/continual_lora.s`, `tools/hybrid_eval.py`, `tests/`: adaptation experiment, probability interpolation and verification harnesses.

## Based on

These are standard techniques: Euclid GCD rational arithmetic; INT8 with per-32-value scale (the same layout family as GGML Q8_0); blocked GEMM; online softmax; n-gram/neural interpolation; RMSNorm; RoPE; GQA. The g32 size and source contracts are in [`evidence/docs-refresh-20261031/contract-excerpts.json`](evidence/docs-refresh-20261031/contract-excerpts.json). The native activation scales and weight packing are not a GGML file format.

| Source | Author | Use | License / record |
| --- | --- | --- | --- |
| [Qwen model](https://huggingface.co/Qwen/Qwen3-8B-Base) | Qwen Team | External pretrained weights and architecture | Apache license; [source register](docs/SOURCES.md) |
| [GGML](https://github.com/ggml-org/ggml) | The ggml authors | Block quantization reference | MIT |
| [NN_S](https://github.com/zenyyxz/NN_S) | zenyyxz | Assembly implementation reference | MIT |
| [matmul_assembly_x86](https://github.com/alint77/matmul_assembly_x86) | alint77 | GEMM worklog reference | No license declared in repository metadata |
| [niyah-lang](https://github.com/Grar00t/niyah-lang) | Suliman Alshammari | Vendored morphology files and external compiler | MIT; `src/morph_tokenizer/upstream/PROVENANCE.json` |
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | Ronen Eldan, Yuanzhi Li | External evaluation data | Dataset license in [`evidence/tinystories-external-32epochs-local.json`](evidence/tinystories-external-32epochs-local.json) |

[SOURCES.md](docs/SOURCES.md) names the papers, authors, URLs and license records for the standard techniques, tools and retained references. Model weights, datasets, dependency packages and vendored files are external work. The repository has no root license; the vendored MIT notice applies to its stated files.

## Limitations

- Teacher-forced agreement measures the recorded inputs. A completed KV-cache generation comparison is absent from this branch.
- Qwen execution requires Python and its dependencies. The native shared library also uses the C runtime, math library and OpenMP.
- The output projection remains FP32. Quantization scales and accumulation include FP64; see [`evidence/docs-refresh-20261031/contract-excerpts.json`](evidence/docs-refresh-20261031/contract-excerpts.json).
- The current byte-model context, buffers and checkpoint/oracle layouts disagree. Historical losses do not establish current compatibility. [`evidence/docs-refresh-20261031/contract-excerpts.json`](evidence/docs-refresh-20261031/contract-excerpts.json); [`evidence/docs-refresh-20261031/verify-all.log`](evidence/docs-refresh-20261031/verify-all.log)
- The full-model reports do not verify official weight-file digests. Their stored configuration and script hashes do not establish upstream weight identity. [`tools/qwen/verify/full_gate_g32all_20261031_133521.json`](tools/qwen/verify/full_gate_g32all_20261031_133521.json)
- GEMM uses shared scratch and requires serial calls. AVX512/AMX paths lack hardware execution evidence on the recorded host. [`evidence/kernel-suite-20261031-112031-843492/results-o2-full/metadata.json`](evidence/kernel-suite-20261031-112031-843492/results-o2-full/metadata.json); [`evidence/nexus-kernels.json`](evidence/nexus-kernels.json)
