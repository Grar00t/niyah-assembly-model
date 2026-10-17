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
128 KiB per input, matching input/test bytes, and output paths pointing at
either input, including **hard links to the same inode**.

**Output safety (PR #9 hard-link correction):** the CLI checks path identity
before beginning training and again before writing, using `os.path.samefile`
on existing outputs. It refuses symlink outputs. A valid report is staged in a
new private temporary file in the output directory, flushed, and atomically
renamed with `os.replace`. It never uses `write_text`/`O_TRUNC` on a
pre-existing output inode, so a hard-linked input is not truncated even by
an alias appearing during an in-place-write race. An output file unrelated
to the inputs may still be replaced; point `--output` at a dedicated report
path. All destructive regression cases use disposable TemporaryDirectory
fixtures only.

**Runtime bounds:** input files each have a separate 128 KiB size cap, but
not all 128 KiB / 100-epoch combinations are accepted. Before native
training, the tool checks the conservative workload proxy
`max(epochs, 1) * train_bytes + test_bytes <= 262144` byte-passes.
The backend still has a fixed 120-second execution deadline: slow or
heavily loaded machines may time out sooner than this budget predicts.
Such a timeout produces a handled error (exit 2), not a Python traceback
or a partial JSON report. Reduce bytes/epochs if rejected or timed out.

These checks are NOT a security sandbox or formal protection against
malicious same-user directory/filename races. Do not execute untrusted
Assembly source. Continue using replaceable copies rather than valuable
original input files until your deployment's directory permissions and
threat model are reviewed.

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

Remediation evidence (disposable red/green tests and full-suite hashes):
[evidence/local-byte-hardlink-safety-20261017.json](../evidence/local-byte-hardlink-safety-20261017.json).
