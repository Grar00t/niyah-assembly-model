# Niyah Assembly Model

Linux x86-64 trigram softmax model in GNU Assembly, without libc. Two context bytes, 64 MiB float32 weights, masked max-shifted softmax, explicit normalization, cross-entropy, SGD, x87 exp/log and Linux syscalls.

## Build and run

```sh
make
make run
make test
```

Requires Linux x86-64, GNU toolchain and make. Tests additionally require Python 3.

## Data

The training corpus is embedded in src/trigram.s. data/corpus.txt is a matching inspection/test copy, not a runtime input. Change both together. Weights initialize to zero and train anew each run; no checkpoints are saved.

## Verification

The packaged tests check generation length/alphabet for four seeds and deterministic replay (5 tests). The user separately ran the extended softmax and Niyah lexer suite: 22/22 passed. That extended runner and lexer executable are not included here. Passing tests do not prove memory safety or language quality.

## Limitations

Tiny demonstration corpus. No BPE trainer, runtime dataset loader, held-out evaluation, checkpointing or UTF-8-constrained sampling. This is not a conversational LLM.
