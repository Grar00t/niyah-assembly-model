# Opt-in 256-byte support and OOV evaluation (2026-10-16)

## Scope

The 9,616-float, eight-byte-context neural model remains native
GNU x86-64 Assembly with direct Linux syscalls and independently learned
Q/K/V parameters. This change is about **evaluation support**, not a new
language-model architecture or a native neural/count hybrid.

The previous evaluation refused TinyStories validation rows 200..207
because seven byte IDs were absent from training:
[45, 48, 49, 50, 51, 56, 57].
They occurred only 8 times among 12,373 scored positions. A masked
model assigned them zero probability. Dropping those positions would
inflate the apparent heldout result.

## Explicit vocabulary policy

| Mode | Next-byte support | Count smoothing | Saved checkpoint |
| --- | --- | --- | --- |
| Default (unchanged) | Training-seen bytes | Additive 0.5 over seen IDs | Existing v2 |
| FULL_BYTE_VOCAB=1 | All 256 byte IDs | Positive alpha over 256 IDs | Existing v2 forbidden |

The opt-in mode assembles src/attention_lm.s with the GNU assembler flag
-Wa,--defsym,FULL_BYTE_VOCAB=1. The tools/hybrid_eval.py function
native_true_probs(..., full_byte_vocab=True) compiles a disposable static
ELF and executes native Assembly SGD/forward/probability export.
The trigram counts and linear interpolation still run in Python **only
for evaluation**; this does not add an Assembly hybrid generator.

In exact arithmetic, all 256 masked-softmax IDs have positive support.
Extreme float32 logits can numerically underflow to zero: the evaluator
rejects zero, negative, NaN, or non-finite exported probabilities.
This is not an assurance of useful OOV prediction or valid UTF-8 output.

The v2 checkpoint manifest does not encode vocabulary policy. For safety,
FULL_BYTE_VOCAB with INFERENCE or EXPORT_WEIGHTS fails at assembly time.
A future full-byte checkpoint needs a versioned contract rather than
silent reinterpretation of weights.

## Local verification

~~~sh
make test-fullbyte-oov
make test-hybrid-eval
make verify-all
~~~

The OOV regression is offline: synthetic unseen digits, a separately
implemented 256-way Counter reference oracle, native true-byte probability
stream and independent NLL reconstruction, exact mixture endpoint checks,
and compile-failure assertions for incompatible checkpoint modes.
CI runs these tests without downloading licensed third-party datasets.

## Locked revisit of the previously refused TinyStories slice

~~~sh
python3 tools/oov_revisit.py --epochs 32
~~~

Source: https://huggingface.co/datasets/roneneldan/TinyStories
License metadata: CDLA-Sharing-1.0. Dataset text is downloaded in memory
and **not committed**. The runner refuses changed row IDs, processed bytes,
SHA-256, or exact duplicate source stories.

- Train: official train rows 0..11, 8,575 bytes,
  SHA256 d284702c7392bedc3373d0a4f0400870d563265c41d479c6a7d9c57904f3dca4
- Calibration: official validation rows 0..5, 4,212 bytes,
  SHA256 ad1d534077e9e29b21c9e4bb912afa05f309bf5d4cedafcdb77c3b16b9660660
- Revisit: official validation rows 200..207, 12,381 bytes,
  SHA256 57c7ceb60e460aaf9d69f3ec58f163f81055195f07268fefe9566ad7f8b0975a

Alpha (256-byte smoothed trigram) and lambda (probability mixture)
were both selected **from calibration rows only**, using fixed grids:
alpha=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0) and
lambda=(0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1).
Selected alpha=0.01 and lambda=0.4; 32 Assembly training epochs.

| Mean next-byte NLL, nats/byte | Calibrated trigram | Assembly attention | Calibrated hybrid |
| --- | ---: | ---: | ---: |
| All 12,373 positions | 2.498354 | 2.535610 | **2.221572** |
| 8 actual OOV targets | **8.860953** | 15.159669 | 9.365569 |

Aggregate hybrid NLL is 11.1% lower than calibrated trigram on this
diagnostic slice, but **OOV-only NLL is worse than trigram**.
The aggregate is overwhelmingly driven by in-vocabulary targets.
Original alpha=0.5 trigram NLL was 3.235971, underscoring the need for
calibration instead of praising a weak strawman baseline.

Machine-produced evidence, loss grids, source hashes and independent
recomputation receipts: evidence/fullbyte-oov-revisit-local.json.

## Limits and next gate

**NOT A BLIND TEST:** rows 200..207 were already inspected and rejected
due to unseen byte IDs before this extension was developed. The
calibration slice was also previously explored. Prior masked-vocabulary
results on validation rows 100..107 are not directly comparable to a
256-byte support contract. Only 8 unseen targets were observed.

No native count-model mixing, general semantic ability, usable UTF-8
generation, production safety, third-party replication, or actual
end-user benefit has been established. The assembly model has only
eight bytes of context and one attention head.

Next gate: declare an entirely new OOV-containing heldout slice and
immutable hashes before inspecting outcomes; measure tuned count
baselines, OOV-only NLL, in-vocabulary NLL and computation cost;
consider a native hybrid generator only if those tests justify it.

Reference: Neubig and Dyer (2016), Generalizing and Hybridizing
Count-based and Neural Language Models, https://aclanthology.org/D16-1124/ .
The fixed lambda used here is not their context-dependent learned mixer.
