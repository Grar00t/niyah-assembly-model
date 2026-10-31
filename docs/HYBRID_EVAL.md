# Attention and trigram interpolation

`tools/hybrid_eval.py` applies standard n-gram/neural probability interpolation to probabilities exported by the native byte model. The mixture is evaluated in Python and is not integrated into native generation.

Source: *Generalizing and Hybridizing Count-based and Neural Language Models*, Graham Neubig and Chris Dyer, [paper](https://aclanthology.org/D16-1124/). Paper license: CC BY 4.0, recorded in [`evidence/docs-refresh-20261031/sources.json`](../evidence/docs-refresh-20261031/sources.json). Dataset: TinyStories by Ronen Eldan and Yuanzhi Li; CDLA-Sharing-1.0 in [`evidence/tinystories-external-32epochs-local.json`](../evidence/tinystories-external-32epochs-local.json).

## Recorded protocol

Training: rows 0..11, 12 documents, 8575 bytes. Calibration: validation rows 0..5, 6 documents, 4212 bytes. Test: validation rows 100..107, 8 documents, 5571 bytes. The calibration grid is [0, .05, .1, .2, .4, .6, .8, 1]; it selects attention weight .4 and trigram weight .6. All source and processed hashes remain in [`evidence/hybrid-fresh-test-local.json`](../evidence/hybrid-fresh-test-local.json).

| Test predictor | NLL, nats/byte | Evidence |
| --- | ---: | --- |
| Native attention | 2.242482512 | [`evidence/hybrid-fresh-test-local.json`](../evidence/hybrid-fresh-test-local.json) |
| Trigram | 2.035952147 | [`evidence/hybrid-fresh-test-local.json`](../evidence/hybrid-fresh-test-local.json) |
| Fixed mixture | 1.975534497 | [`evidence/hybrid-fresh-test-local.json`](../evidence/hybrid-fresh-test-local.json) |
| Mixture minus trigram | -0.06041765 | [`evidence/hybrid-fresh-test-local.json`](../evidence/hybrid-fresh-test-local.json) |

The mixture loss is 2.97% lower than trigram after rounding. The input fields and calculation are in [`evidence/docs-refresh-20261031/derived-measurements.json`](../evidence/docs-refresh-20261031/derived-measurements.json).

All calibration losses and independent native-probability reconstruction results remain in the same receipt. This historical experiment does not validate the changed byte-model source.

## Preserved failed replication

Validation rows 200..207 contain unseen byte values [45, 48, 49, 50, 51, 56, 57]. The processed slice contains 12381 bytes. Status is `BLOCKED_ON_OOV_NOT_SCORED`; the slice was not substituted and no loss was computed. [`evidence/hybrid-oov-preflight-local.json`](../evidence/hybrid-oov-preflight-local.json)

```sh
make test-hybrid-eval
python3 tools/hybrid_eval.py --epochs 32
```

The epoch count comes from [`evidence/hybrid-fresh-test-local.json`](../evidence/hybrid-fresh-test-local.json). The offline tests use fixtures; the experiment fetches dataset rows. The training-byte vocabulary mask prevents arbitrary-byte evaluation. Distinct row IDs do not exclude semantic overlap, and lower loss on this slice does not measure user-facing generation quality.
