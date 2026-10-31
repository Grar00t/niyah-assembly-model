# Qwen validation runtime

Python orchestrates and loads weights; C computes norm, RoPE, attention, scaling; Assembly computes INT8 GEMM.

`tools/qwen/qwen_native_full_gate.py` imports the native source strings from `tools/qwen/qwen_native_layer0_g32_tany.py`, compiles a shared library and calls it through `ctypes`. Python also performs weight quantization and prepares the reference tensors and rotary tables. The C attention loop calls an Assembly floating dot kernel. The output projection uses C intrinsics and OpenMP. [`evidence/docs-refresh-20261031/contract-excerpts.json`](../evidence/docs-refresh-20261031/contract-excerpts.json)

## Numerical contract

Weights use a symmetric scale per output row. Activations use a scale per 32 values in g32 mode. Scales and group accumulation are FP64; buffers are FP32; integer products accumulate in INT32. Rounding is ties-to-even and values are clipped to the signed symmetric range. These are standard block-quantization techniques, in the same layout family as GGML Q8_0; the scale representation and row-scaled weights differ from that format. [`evidence/docs-refresh-20261031/contract-excerpts.json`](../evidence/docs-refresh-20261031/contract-excerpts.json)

The recorded comparison includes 36 decoder layers, final RMSNorm and `lm_head`, with 10 prompts (5 Arabic, 5 English) and 32 positions per prompt. The reference loads each decoder layer in FP32; it is not a single `model.forward` call. [`tools/qwen/verify/full_gate_g32all_20261031_133521.json`](../tools/qwen/verify/full_gate_g32all_20261031_133521.json)

| Mode | Top-1 agreement | Mean KL, nats | Top-5 overlap | Gate | Evidence |
| --- | ---: | ---: | ---: | --- | --- |
| g32 at every linear | 0.959375 | 0.005304766440675167 | 0.958125 | PASS | [`tools/qwen/verify/full_gate_g32all_20261031_133521.json`](../tools/qwen/verify/full_gate_g32all_20261031_133521.json) |
| Per-token; g32 down projection | 0.853125 | 0.6338464501131247 | 0.875625 | FAIL | [`tools/qwen/verify/full_gate_pertoken_20261031_132909.json`](../tools/qwen/verify/full_gate_pertoken_20261031_132909.json) |

Gate: top-1 agreement at least 0.95 and mean KL at most 0.03 nats. Top-5 overlap and per-layer relative error are informational. [`tools/qwen/verify/full_gate_g32all_20261031_133521.json`](../tools/qwen/verify/full_gate_g32all_20261031_133521.json)

Peak process RSS is 3772.6 MiB for g32 and 3871.9 MiB for the per-token run. The JSON field is named `peak_rss_mb`; the function divides Linux `ru_maxrss` KiB by 1024. These totals include the reference harness. [`tools/qwen/verify/full_gate_g32all_20261031_133521.json`](../tools/qwen/verify/full_gate_g32all_20261031_133521.json); [`tools/qwen/verify/full_gate_pertoken_20261031_132909.json`](../tools/qwen/verify/full_gate_pertoken_20261031_132909.json); [`evidence/docs-refresh-20261031/contract-excerpts.json`](../evidence/docs-refresh-20261031/contract-excerpts.json)

## Reproduction

Use an existing Python environment with `numpy`, `torch`, `safetensors` and `transformers`; use GCC with OpenMP and an AVX2 CPU. The script reads local weights with remote-code loading disabled. Its GEMM source path is hard-coded to `$HOME/niyah-assembly-model/src/niyah_i8_gemm.s`; place the checkout there or account for that dependency when reproducing. The reported source hash must match the intended source.

```sh
export NIYAH_MODEL_DIR="$HOME/models/Qwen3-8B-Base"
export NIYAH_REPORT="$(mktemp -d)/g32.json"
NIYAH_G32_ALL=1 python3 tools/qwen/qwen_native_full_gate.py
export NIYAH_REPORT="$(mktemp -d)/pertoken.json"
NIYAH_G32_ALL=0 python3 tools/qwen/qwen_native_full_gate.py
```

The model and switches above correspond to [`tools/qwen/verify/full_gate_g32all_20261031_133521.json`](../tools/qwen/verify/full_gate_g32all_20261031_133521.json) and [`tools/qwen/verify/full_gate_pertoken_20261031_132909.json`](../tools/qwen/verify/full_gate_pertoken_20261031_132909.json). Inspect `result.gate` after execution. A FAIL in that field does not currently set a nonzero process exit.

## Decode and remaining limits

`tools/qwen/qwen_native_decode_kv.py` exists in the local development checkout as untracked work. No finished decode report was present in the inspected verification directory. Status: in progress. The untracked script is not included in this documentation branch. [`evidence/docs-refresh-20261031/inspection.json`](../evidence/docs-refresh-20261031/inspection.json)

The gate reports do not verify official model-file digests. The kernel uses shared scratch and is called serially. The comparison does not establish generation quality, concurrent execution or throughput outside the recorded workload. [Sources](SOURCES.md).
