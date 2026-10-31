# Arabic surface codec

`src/morph_tokenizer/tokenizer.c` wraps the vendored niyah-lang morphology analyzer. It preserves exact input bytes, including invalid encoding, whitespace and punctuation. Candidate boundaries select a surface encoding; roots do not replace input text.

```sh
make morph-tokenizer test-morph-tokenizer
build/morph-tokenizer encode < input.txt > input.nymt
build/morph-tokenizer decode < input.nymt > reconstructed.txt
build/morph-tokenizer vocab
```

The recorded vocabulary has 294 IDs. Byte fallback uses the byte-valued IDs; the remaining entries are exact surface strings in `src/morph_tokenizer/vocab.json`. The recorded checks include 91 round trips and 18 malformed cases. [`evidence/nexus-morph-tokenizer.json`](../evidence/nexus-morph-tokenizer.json)

The binary stream binds the vocabulary version and digest. Decode validates the stream before writing output. Limits, header sizes and exit contracts are captured from source in [`evidence/docs-refresh-20261031/contract-excerpts.json`](../evidence/docs-refresh-20261031/contract-excerpts.json). A digest binds ID meanings, not authenticity.

Vendored files are `src/morph_tokenizer/upstream/morph.c`, `morph.h` and `lexer.h`. Author: Suliman Alshammari. Source: [niyah-lang](https://github.com/Grar00t/niyah-lang). License: MIT, retained beside those files. `PROVENANCE.json` records source identity and file hashes. [Source register](SOURCES.md).

The byte-model checkpoint cannot be interpreted as this token model. Fixture byte reduction does not measure Arabic accuracy, BPE performance or KV-cache reduction. [Token-model experiment](MORPH_ATTENTION.md).
