# Assembly + Trigram hybrid: held-out experiment (2026-10-16)

## Goal and source

Check whether the current 9,616-parameter eight-byte-context NIYAH
Assembly model adds predictive information beyond smoothed trigram.

Research precedent: Graham Neubig and Chris Dyer (2016),
https://aclanthology.org/D16-1124/
The interpolation concept is **not claimed as a new invention**.

Dataset: https://huggingface.co/datasets/roneneldan/TinyStories
License metadata: CDLA-Sharing-1.0. No original dataset text is committed.

## Native and external components

- Assembly switch: -Wa,--defsym,DUMP_VALID_PROBS=1
- Native executable runs 32 SGD epochs using training bytes only,
  reports final validation NLL, outputs VALID_TRUE_PROBS_F32 followed
  by exactly one little-endian float32 probability per true validation
  next byte, then exits. It is static ELF with Linux syscalls, no libc.
- Python harness checks the float stream reproduces Assembly NLL within
  0.001 nats/byte and implements the count-based trigram/mixing layer.
- Trigram uses alpha=0.5 additive smoothing over the training alphabet,
  with counts exclusively from training bytes.
- p_mix = (1 - lambda) * p_trigram + lambda * p_attention
- All models score validation positions 8..N-1 with NLL nats/byte.
- The hybrid is **not yet integrated into the native generator**.
  No claim of an end-to-end purely Assembly hybrid.

## Frozen protocol

Training: TinyStories train 0..11, 12 stories / 8575 lowercase bytes.
Training SHA256:
d284702c7392bedc3373d0a4f0400870d563265c41d479c6a7d9c57904f3dca4

Calibration: TinyStories validation 0..5, 6 stories / 4212 bytes.
This previously measured slice IS NOT an unseen final test.
Calibration SHA256:
ad1d534077e9e29b21c9e4bb912afa05f309bf5d4cedafcdb77c3b16b9660660

Grid searched on calibration ONLY: 0, .05, .1, .2, .4, .6, .8, 1.
Selected attention mixture weight = 0.4; trigram weight = 0.6.

Fresh test: TinyStories validation 100..107, 8 stories / 5571 bytes.
The slice and processed SHA were recorded before any test loss
was computed. No mixture retuning on this slice.
Fresh test SHA256:
13140d20d59ed4aebd0bc088a0fdb78fe8292eb51e39230d35ae4b7126ae97c1

| Predictor | NLL nats/byte on fresh test |
| --- | ---: |
| Trigram alone | 2.035952147 |
| Native standalone Attention | 2.242482512 |
| Fixed 0.4 Attention + 0.6 Trigram | **1.975534497** |

Hybrid vs trigram: -0.060417650 nats/byte, approximately 2.97% lower.
Source story hashes and complete calibration loss grid are preserved
in evidence/hybrid-fresh-test-local.json. The native output matches
Python's independent NLL reconstruction on both calibration and test.

## Attempted replication / explicit failure

Another predeclared slice, validation 200..207, contains unseen byte
values [45, 48, 49, 50, 51, 56, 57] (hyphen and several digits).
Those bytes do not appear in the training alphabet. Both predictors'
current probability-support contract prohibits fair direct comparison;
the experiment **failed closed without scoring or substituting** this
slice. No generalization to arbitrary bytes is claimed.

Replication processed-text SHA256:
57c7ceb60e460aaf9d69f3ec58f163f81055195f07268fefe9566ad7f8b0975a
Length: 12381 bytes. This cannot count as a passing replication.

## Commands

Offline unit/integration tests:
    make test-hybrid-eval

Retrieve pinned official dataset rows and rerun the native experiment:
    python3 tools/hybrid_eval.py --epochs 32

The offline fixture tests native Assembly probability stream integrity,
independent trigram oracle, zero/one mixture boundaries, out-of-vocabulary
failure and invalid weight rejection. GitHub CI deliberately does not
download licensed third-party training data.

## Evidence limits

- One fixed seed, tiny training data, 8 heldout documents; this is an
  exploratory architecture demonstration, not proof of broad efficacy.
- Distinct row IDs and hashes do not rule out semantic near-duplicates.
- Loss improvement is not proof of useful generation or human benefit.
- Conditional distributions are normalized on training-seen byte IDs
  only. Vocabulary expansion must be independently tested on a newly
  sealed OOV-inclusive dataset.
- There was no GPU rental, MSVC cl.exe or Python neural training.

NEXT GATE: predefine a shared 256-byte positive support for both models,
separate selection and truly fresh OOV-inclusive test, then verify
predictive gain and memory/latency costs before integrating the
count-based sidecar into a production Assembly predictor.
