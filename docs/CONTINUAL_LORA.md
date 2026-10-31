# Projection adaptation experiment

`src/continual_lora.s` is a static Linux executable. It applies standard low-rank adaptation to the frozen byte model's WQ projection. It does not attach an adapter to Qwen or to the native text generator.

The projection is `y = WQ*x + B*(A*x)`. The source declares rank 2 and 64 trainable adapter parameters. Its base contract is the historical 38464-byte checkpoint; the persisted adapter includes a complete base snapshot and totals 38792 bytes. These are file-format constants, not executed measurements. [`evidence/docs-refresh-20261031/contract-excerpts.json`](../evidence/docs-refresh-20261031/contract-excerpts.json)

```sh
make continual-lora
make test-continual-lora
```

The existing test builds its base from `src/attention_lm.s` unless `NIYAH_LORA_BASELINE` selects a preserved source. The current byte-model layout mismatch applies to that default. A compatible base is required; a build alone is not evidence of successful adaptation. [Byte-model status](ATTENTION.md).

Updates use old adapter values for both derivatives, clip gradients and evaluate supplied new/replay records. Accepted output uses a new path with exclusive creation. The input adapter is not opened for writing. Rejected candidates do not replace it. A caught output failure removes the new partial output created by that invocation; interruption can still leave an incomplete file.

The file binds the base by byte equality and uses FNV for accidental-corruption detection. It is not a signature. Replay records are training inputs; the replay bound does not establish retention on unseen tasks.

`tests/test_continual_lora.py` contains analytic and finite-difference oracles, persistence, rejection, numeric bounds and malformed-file checks. No finished execution receipt was found in the inspected evidence directories. [`evidence/docs-refresh-20261031/inspection.json`](../evidence/docs-refresh-20261031/inspection.json)

Source technique: *LoRA: Low-Rank Adaptation of Large Language Models*, Edward J. Hu and coauthors. Paper URL, complete authors and license record are in [SOURCES.md](SOURCES.md).
