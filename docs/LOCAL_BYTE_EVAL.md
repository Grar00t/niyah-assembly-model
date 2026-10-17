# Offline user-file scoring using x86-64 Assembly

## Use

For developers and students who want a reproducible local byte-language-model
evaluation without uploading their own input data.

Requires Python 3, GNU cc/as, Linux/WSL x86-64. No network or pip dependencies.
The attention model trains/exports probabilities as native Assembly;
the count model and static mixture evaluate in Python.

~~~sh
python3 tools/local_byte_eval.py --train path/to/train.bin --test path/to/test.bin --epochs 2 --alpha 0.01 --lambda-weight 0.4 --output local-report.json
make test-local-byte-eval
make verify-all
~~~

The output reports input SHA-256 hashes, lengths, scored byte positions, OOV
byte IDs and target counts, native loss consistency and full/OOV-only negative
log likelihoods for neural Assembly, smoothed trigram and Python mixture.
Lower loss is better; OOV-only losses are null if there are no OOV targets.

This program does not make external connections or include input content in
its JSON. It uses temporary local copies of source and dataset files. It
rejects nonregular files, symlink paths, less than nine bytes, file sizes above
128 KiB per input, matching input/test bytes and output paths pointing at
either input. These are safety checks, NOT a security sandbox. Do not execute
untrusted Assembly source. TOCTOU races or malicious path trees are beyond
this local teaching-tool contract.

Default alpha=0.01 and lambda=0.4 came from a TinyStories experiment, not
calibration on each user's data. Test on independently collected material;
different file bytes do not guarantee test independence. The tool measures
next-byte probability, not useful text output or security. A byte is neither
a Unicode character nor an assembly instruction.

The repo currently lacks a user-chosen code license; redistribution to a
Hugging Face project requires a declared license and HF authentication.
No real-world user-value measurement has yet occurred.

Research:
- ASMA-Tune (2025), https://arxiv.org/abs/2503.11617
- RL Assembly optimization (2025), https://arxiv.org/abs/2505.11480
- Count/neural mixtures, https://aclanthology.org/D16-1124/
