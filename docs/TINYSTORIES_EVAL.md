# TinyStories: real external held-out measurement

This experimental protocol benchmarks the existing **9,616-parameter,
8-byte-context**, trainable Q/K/V **x86-64 Assembly next-byte model**.
It does NOT establish large-model language understanding or useful generation.

## Public data and reproducibility

Dataset: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
announced license **CDLA-Sharing-1.0**, paper
[TinyStories (Eldan and Li)](https://arxiv.org/abs/2305.07759).

Fetch exactly train rows 0..11 and validation rows 0..5 using the HF
Dataset Viewer API (/rows, config=default). The two original HF splits
are distinct; the harness also rejects duplicate exact story SHA-256 hashes
and unsupported validation bytes. Preprocess both identically: concatenate
story texts with two LF separators, append LF, UTF-8 encode, and lowercase
ASCII bytes. No truncation, shuffle or sample selection based on score.
The 12 training documents comprise **8,575** encoded bytes and the 6 heldout
documents **4,212** bytes. All validation byte IDs appear in training.

**No TinyStories corpus bytes are committed.** Evaluation copies source
code and training/validation data into a fresh temporary build directory.
The pure-Assembly native ELF contains no Python, libc, API calls, or
pretrained weights. Only the evaluation harness uses Python/network.

Run locally from repository root:

```bash
python3 tools/evaluate_tinystories.py --epochs 8 --save evidence/tinystories-external-8epochs-local.json
python3 tools/evaluate_tinystories.py --epochs 32 --save evidence/tinystories-external-32epochs-local.json
python3 tests/test_tinystories_eval.py
```

The remote data and results are **not fetched in CI**: CI runs a
network-free synthetic regression with no external dataset license risk.
Scores are mathematically mean next-byte NLL in **nats/byte**.
All methods evaluate indices 8..N-1 on the same heldout bytes.
The baseline smoothing is symmetric Laplace/Dirichlet alpha=0.5 over
the training-byte vocabulary. Original 2-byte lookup is a trigram
*statistical* baseline, not the 64 MiB legacy Assembly implementation
retrained on this corpus. No synthetic prompts were counted as results.

## Measured October 16, 2026, DESKTOP-2P1JAIV WSL

| Predictor | Validation NLL (nats/byte) |
| --- | ---: |
| Native attention, 0 epochs | 3.538 |
| Native attention, 8 epochs | 2.317 |
| Native attention, 32 epochs | 2.315 |
| Smoothed unigram (no context) | 2.963912 |
| Smoothed bigram (1-byte context) | 2.300459 |
| Smoothed trigram (2-byte context) | **2.068353** |

The 32-epoch attention model trails smoothed trigram by **0.246647
nats/byte** on the fixed heldout slice. Train loss drops from 2.089
to 1.989 between 8 and 32 epochs, but heldout barely improves
(2.317 to 2.315). This is **negative evidence** for a benefit from
additional training under this setup, not a reason to cherry-pick
longer runs.

### Important limitations

- Only 12/6 relatively similar stories; independent splits and exact
  text hashes do not prove lack of semantic or near-duplicate overlap.
- Model has only 8 byte positions of context and lacks a full Transformer
  decoder. The relative win of a smoothed trigram is meaningful but
  is not proof about larger attention models.
- Byte normalization is declared; this is NOT the paper's original
  tokenizer, architecture or benchmark. No end-user benefit measured.
- The heldout corpus is embedded in the training binary for loss
  *evaluation*, but gradients use train_bytes only; no teacher leakage
  into updates. Repeat on separate training and evaluation processes
  for higher assurance.
- End-to-end network reproducibility is subject to dataset revisions;
  the receipt records complete hashes for the chosen source documents
  and processed streams. Fail if hashes change in a future rerun.
- 32 epochs are not enough to establish the best achievable model
  with improved scheduling, context, capacity or data size.

Next justified engineering experiment: increase context length to
32 with an explicit buffer/ISA audit, retrain on the same fixed
heldout protocol, and require a lower heldout loss than **2.068353**
before claiming benefit. GPU purchases are not warranted from
these results.
