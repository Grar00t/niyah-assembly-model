# Native continual LoRA experiment

`build/continual_lora` is a separate static Linux x86-64 executable. It uses
GNU Assembly, scalar SSE arithmetic, and raw Linux syscalls. Runtime needs
no libc, Python, CUDA, PyTorch, shared libraries or model service. GNU tools
build it; the independent test harness uses Python's standard library.
Linux/WSL remains an operating-system dependency: this is not a bootable OS.

## What is adapted

The executable loads the existing 38,464-byte native attention checkpoint
layout and freezes its learned 16x16 `WQ` projection at byte offset 35,392.
Given a supplied 16-float feature vector `x`, it evaluates

```
h = A x                 A: 2 x 16
y = WQ x + B h          B: 16 x 2
L = 1/2 sum_j (y_j-target_j)^2
grad B = (y-target) h^T
grad A = [B^T (y-target)] x^T
```

The effective LoRA scaling is `alpha/r = 1`. Only the 64 parameters in A/B
change; the base file and the complete base bytes in memory stay unchanged.
Both derivatives use the pre-update A/B values. Initialization gives A
deterministic nonzero rows and B zeros, so the initial adaptation is zero.

This is an independently runnable projection experiment. It does not insert
the adapter into attention generation, replace the model's eight-byte
working context, establish a personality, or demonstrate Arabic quality.
The tests train a native base in a private temporary directory and adapt
synthetic 16D tasks. They also verify forward output on a real learned
embedding row. No existing model file is overwritten.

## Commands

From the repository, include `Makefile.lora` in the main Makefile or run:

```sh
make -f Makefile.lora continual-lora
make -f Makefile.lora test-continual-lora
./build/continual_lora init base.f32 adapter-0.lora
./build/continual_lora predict base.f32 adapter-0.lora features.f32 > prediction.f32
./build/continual_lora update base.f32 adapter-0.lora new.f32 replay.f32 adapter-1.lora 128
```

`predict` consumes exactly 16 little-endian float32 values (64 bytes) and
emits exactly 16 float32 values. `new.f32` and `replay.f32` each contain
1..16 records. A record is 16 features followed by 16 targets: 128 bytes.
All values must be finite. Features have absolute bound 4, targets 64,
base parameters 64 and adapter parameters 8. Inputs beyond these declared
limits are rejected rather than silently clipped or reinterpreted.

Every update performs 1..256 epochs, each consisting of one pass over new
records followed by one pass over replay records. Every gradient component
is clipped to [-1,1], and the learning rate is fixed at 1/32. A parameter
step is therefore bounded by 1/32. Any parameter leaving [-8,8] rejects
the entire candidate.

Promotion additionally requires:

* Mean half squared error on the new records improves by more than 1e-7.
* Mean half squared error on replay records rises by at most 0.005.

These are explicit bounds on the supplied replay records only. They do not
prove absence of forgetting on unseen tasks. Replay records are training
data in this experiment, not an independent generalization benchmark.
The arithmetic oracle is independent; a symbolic calculator does not certify
neural retention or language quality.

## Persistence and rejection

An accepted update writes a NEW adapter path with exclusive creation and
no symlink following, synchronizes it with `fsync`, and closes it. A
pre-existing output path is rejected. The input adapter is never opened
for writing. A gate/numeric rejection restores in-memory A/B and creates
no output file. A caught output failure removes only the file just created
by this invocation. Power loss can still leave an incomplete new file;
strict load validation rejects it. This is not an atomic active-pointer
switch, cross-process lock, or crash-recovery journal.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Operation completed; update candidate passed its declared gates |
| 2 | Invalid command, size, checkpoint identity, numeric data or format |
| 3 | Input/output failure, symlink refusal, or existing output path |
| 10 | Candidate rejected by numeric bounds, improvement or retention gate |

On accepted `update`, stdout is 16 binary bytes: four float32 values in
order `new_before, new_after, replay_before, replay_after`. Diagnostics go
to stderr. Redirect stdout to a file for inspection; it is not text.

## Version-1 file layout

The format deliberately binds the complete small base by **byte equality**.
It copies that base into the checkpoint, increasing storage. It does not
pretend a 256-byte parameter update is the complete persisted artifact.

| Offset | Bytes | Meaning |
|---:|---:|---|
| 0 | 8 | Magic `NLORA001` |
| 8 | 4 | Version = 1 |
| 12 | 4 | Dimension = 16 |
| 16 | 4 | Rank = 2 |
| 20 | 4 | Base bytes = 38,464 |
| 24 | 4 | Parameter count = 64 |
| 28 | 4 | Reserved, must be zero |
| 32 | 8 | Accepted update count |
| 40 | 8 | Total native SGD record updates |
| 48 | 16 | Reserved, must be zero |
| 64 | 38,464 | Exact complete base checkpoint snapshot |
| 38,528 | 128 | A, row-major float32 |
| 38,656 | 128 | B, row-major float32 |
| 38,784 | 8 | FNV-1a-64 of all preceding bytes, little-endian |

Total: **38,792 bytes**. FNV is only an accidental-corruption check and is
not cryptographic authentication. Runtime checks exact length, format,
reserved fields, finite parameter bounds, checksum, and complete base byte
identity. The harness separately records SHA-256 hashes for evidence.
Base bytes must come from a trusted native checkpoint; this raw legacy
layout does not carry architecture metadata of its own.

## Independent verification

```sh
NIYAH_LORA_BASELINE=/absolute/path/to/preserved_attention_lm.s \
NIYAH_LORA_RECEIPT=/absolute/path/to/lora_receipt.json \
python3 tests/test_continual_lora.py
```

The optional baseline variable selects the preserved source for the private
native base trainer. Otherwise the current `src/attention_lm.s` is used.
The harness never writes `models/attention_weights.f32`.

Tests cover: static ELF dependency inspection; initial frozen projection;
all 64 native gradients against an independent analytic oracle; selected
central finite differences; exact float32 update arithmetic; gradient
clipping; sequential adaptation with replay and independent loss checks;
byte-identical predictions across persisted reload; conflicting-task
rejection; unchanged previous adapters; full-base mismatch away from WQ;
malformed/truncated/extended and corrupted checkpoints; NaN/Inf and range
rejection; record/epoch bounds; output collision and symlink refusal;
parameter-bound rollback; absence of the gradient testing command from the
production executable. The testing command is compiled only when the
assembler symbol `LORA_TEST_STEP` is explicitly defined.
