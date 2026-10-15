# Niyah Assembly Model

Linux x86-64 GNU Assembly byte-trigram softmax model without libc. Two-byte context, 64 MiB float32 weights, max-shifted masked softmax, SGD and direct syscalls.

Commands: make; make run; make test; make checkpoint; make verify; make generate; make test-checkpoint.

Edit data/corpus.txt. The corpus is embedded at build time with .incbin. Checkpoint tooling uses Python 3, validates UTF-8 input and finite weights, and binds source/corpus/weights with SHA-256. Generation verifies and embeds saved weights without retraining. models/ is excluded from Git. Checkpoints do not save optimizer state. Individual file replacements are atomic, not the weights/manifest pair; concurrent writers are unsupported. Checksums are not signatures.

GitHub Actions runs build, generation tests and checkpoint replay. The packaged generation suite has five tests; the earlier external 22-test Niyah/softmax suite is not included. No CI success or memory-safety proof is implied. No BPE, held-out evaluation, hidden neural layer, or UTF-8-constrained generation. This is a small experimental model, not a conversational LLM.
