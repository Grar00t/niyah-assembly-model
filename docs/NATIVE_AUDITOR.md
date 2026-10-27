# Native arithmetic candidate auditor

`nexus_audit` connects a candidate string, including a neural model's
candidate, to the repository's bounded exact rational calculator. It is a
static x86-64 Linux executable written in GNU Assembly. Runtime uses raw
syscalls, with no libc, shell, Python, model service, CUDA or network.
It requires the repository's `exactcalc` executable beside it.

```sh
make nexus-audit test-nexus-audit
./build/nexus_audit add 1 10 2 10 3/10
# VERIFIED 3/10  (exit 0)
./build/nexus_audit add 1 10 2 10 4/10
# MISMATCH       (exit 1)
```

Arguments are operation, numerator1, denominator1, numerator2,
denominator2, and candidate. Candidates must exactly match the calculator's
canonical reduced `N/D` output without its final newline. Thus `0.3`,
`6/20`, and prose containing `3/10` are not accepted as `3/10`. The caller
must explicitly extract a candidate and supply the intended arithmetic
expression. The auditor does not infer that expression from a conversation.

The calculator supports `add`, `sub`, `mul`, `div`, signed32 numerators and
positive denominators <=2147483647. See `EXACT_MATH.md` for the full domain
and checked-intermediate contract.

The auditor obtains its actual location through `/proc/self/exe`, constructs
the sibling `exactcalc` path, and uses `pipe`, `fork`, `dup2`, `execve` and
`wait4`. It does not search PATH or run a shell. It requires a successful
child exit and bounded canonical output framing before comparing bytes.
Pipe reads and writes handle partial transfers and EINTR. SIGPIPE is
ignored so broken stdout is an observable I/O error rather than a signal.
The child inherits this handling. Both binaries and their directory are
assumed trusted; choosing a fixed path does not cryptographically
authenticate a replaced executable. Release receipts hash both binaries.

| Exit | Meaning |
|---:|---|
| 0 | Canonical candidate equals the successful calculator result |
| 1 | Candidate mismatch or noncanonical representation |
| 2 | Incorrect auditor argument count |
| 3 | Calculator/exec failure, rejected calculator input, or pipe/protocol error |
| 4 | Output write failure, including full filesystem or broken pipe |

This proves only a bounded rational arithmetic equality. It makes no claim
about general factual accuracy, language quality, intention, neural memory,
or protection against catastrophic forgetting. It does not automatically
alter a neural model's weights or replace a rejected candidate.

`tests/test_nexus_audit.py` independently computes reference answers with
Python's standard-library `Fraction`. Tests include extreme input bounds,
wrong and neural-style prose candidates, division by zero and invalid
inputs, an empty PATH, a directory containing spaces, a missing sibling,
malformed disposable child executables, `/dev/full`, broken pipes, and
ELF inspection for absent dynamic-interpreter/shared-library dependencies.
Set `NIYAH_AUDITOR_RECEIPT` to save commands, native exit codes and SHA-256
evidence as JSON. These Python tests are build/verification tools; the
delivered calculator and auditor execute natively without them.
