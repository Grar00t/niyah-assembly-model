# TinyStories byte-model evaluation

This is a historical evaluation of the Assembly byte attention model. Its architecture has 9616 parameters and an 8-byte context. The current source differs; see [ATTENTION.md](ATTENTION.md). [`evidence/tinystories-external-32epochs-local.json`](../evidence/tinystories-external-32epochs-local.json)

Dataset: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), Ronen Eldan and Yuanzhi Li. License: CDLA-Sharing-1.0, recorded in [`evidence/tinystories-external-32epochs-local.json`](../evidence/tinystories-external-32epochs-local.json). Paper and license records: [SOURCES.md](SOURCES.md).

## Protocol

Training uses original train rows 0..11: 12 stories and 8575 processed bytes. Validation uses original validation rows 0..5: 6 stories and 4212 bytes. Processing concatenates whole stories with 2 LF separators, encodes UTF-8 and lowercases ASCII. Exact document hashes and processed-stream hashes are recorded. [`evidence/tinystories-external-32epochs-local.json`](../evidence/tinystories-external-32epochs-local.json)

Every method scores positions 8..N-1. The statistical baselines count training data; they are not the legacy Assembly table model retrained on this corpus. The model's training-byte mask covers the recorded validation alphabet. [`evidence/tinystories-external-32epochs-local.json`](../evidence/tinystories-external-32epochs-local.json)

| Predictor | Validation NLL, nats/byte | Evidence |
| --- | ---: | --- |
| Attention, untrained | 3.538 | [`evidence/tinystories-external-32epochs-local.json`](../evidence/tinystories-external-32epochs-local.json) |
| Attention, 8 epochs | 2.317 | [`evidence/tinystories-external-8epochs-local.json`](../evidence/tinystories-external-8epochs-local.json) |
| Attention, 32 epochs | 2.315 | [`evidence/tinystories-external-32epochs-local.json`](../evidence/tinystories-external-32epochs-local.json) |
| Unigram | 2.963911722800764 | [`evidence/tinystories-external-32epochs-local.json`](../evidence/tinystories-external-32epochs-local.json) |
| Bigram | 2.300459449375779 | [`evidence/tinystories-external-32epochs-local.json`](../evidence/tinystories-external-32epochs-local.json) |
| Trigram | 2.068352939196024 | [`evidence/tinystories-external-32epochs-local.json`](../evidence/tinystories-external-32epochs-local.json) |

The neural model loses to trigram by 0.246647 nats/byte. Training loss changes from 2.089 to 1.989 between the recorded runs. [`evidence/tinystories-external-8epochs-local.json`](../evidence/tinystories-external-8epochs-local.json); [`evidence/tinystories-external-32epochs-local.json`](../evidence/tinystories-external-32epochs-local.json)

## Reproduction and limits

```sh
make test-tinystories-eval
python3 tools/evaluate_tinystories.py --epochs 8
python3 tools/evaluate_tinystories.py --epochs 32
```

The epoch arguments match the linked receipts. The harness builds the current source; these commands are not a promise to reproduce a historical score with changed source/data. Use the recorded source/data identity when comparing runs. Do not overwrite existing receipts with `--save`.

Original dataset bytes are not committed. The offline fixture does not download data. Distinct rows and exact hashes do not exclude semantic duplicates. Loss is per processed byte, not the paper's original tokenizer metric. The validation stream is embedded for evaluation; gradient updates use the training stream. Generalization beyond the recorded slice is unmeasured.
