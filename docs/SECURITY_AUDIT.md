# Security review — Niyah native Assembly model

Date: 2026-10-15. Scope: native `src/exactcalc.s`,
`src/attention_lm.s`, `src/trigram.s`, checkpoint and test tools,
and CI. This is a bounded engineering review, **not a claim of
zero vulnerabilities, a penetration test, a formal proof, or certification**.

## Baseline evidence

`make verify-all` passed on the authorized Windows WSL device before
changes. It covered existing trigram and QKV model behavior,
a Python-independent forward oracle, *sample* QKV finite-difference
gradient checks, AVX2 versus scalar behavior, exact checkpoint replay,
source/corpus/weight SHA-256, and disposable corruption negative tests.

The existing experimental model is **9,616 float32 parameters, 8-byte

## Concrete vulnerability fixed: file verification-to-use race

**Before:** `tools/checkpoint_attention.py:generate` called `verify()`
to validate mutable `models/attention_weights.f32` and associated source.
A later assembler invocation in repository cwd used `.incbin` to read
that mutable file again. A concurrent modification between these two
operations could enter the inference executable without matching the
previous hash verification: a classic *time-of-check/time-of-use* gap.

**After:** `verified_inference_snapshot()` copies the exact Assembly
source, two corpus files and 38,464-byte weights into a private temporary
directory. It hashes the **copied bytes** against the versioned manifest,
checks all weights finite, then compiles the snapshot using that cwd.
Modifications to the original mutable path after snapshot cannot
change the built inference binary.

Regression cases on **disposable copies only**:

1. Mutate original weights after snapshot, immediately before assembler:
   output matches untampered inference.
2. Mutate original between initial `verify()` and snapshot creation:
   snapshot checksum mismatch **fails closed**.
3. Compare production weight hash before/after: unchanged.

Both attacks were detected or isolated successfully in local tests.
Limits: same-UID attackers with write access to private sandbox or the
unsigned manifest are not fully excluded; a SHA-256 manifest is an
integrity check, not authenticity. Checkpoint writes of weights and
manifest are separate atomic replacements, not one transaction, so
interrupted training can leave a mismatched pair that requires
regeneration; `verify()` rejects such mismatches.

## Exact arithmetic hardening

- Arithmetic inputs restricted to signed32 numerators and positive
  31-bit denominators, with at most ten ASCII digits per argument.
- Mathematically bounded signed64 intermediates and additional `jo`
  overflow checks. Euclidean GCD yields canonical fractions.
- Zero divisor and parser errors have nonzero exit codes, stdout empty.
- Linux ELF has no runtime loader/undefined imports, and no executable
  stack. Only sys_write and sys_exit are used; no files or sockets.
- Property tests: **5,897 exact successes**, **24 expected rejections**.

Residual risks include untested OS/CPU/loader variants, same-user
tampering, arbitrary-precision requests beyond the declared domain,
side channels, and possible assembly defects not caught by selected
random tests.

## What is and is not verified

| Control | Current evidence | Remaining work |
| --- | --- | --- |
| Scalar/AVX2 neural model math | Existing independent oracles PASS | More training sizes, validation corpus, ISA variation |
| Learnable Q/K/V gradients | Sample finite differences PASS | More coordinates/shapes, exhaustive gradient bounds |
| Inference checkpoint TOCTOU | Reproduction tests PASS after fix | Same-UID attacker model, signed checkpoint identity |
| Exact fraction math | Bounded Fraction differential tests PASS | Broader formal verification, cross-platform tests |
| Niyah build security | Static ELF/NX check PASS | Supply-chain action pinning and independent audit |
| Drone actuator software | NOT PRESENT | Hazard analysis, simulator, independent IV&V |

[NIST SP 800-218 secure development](https://nvlpubs.nist.gov/nistpubs/specialpublications/nist.sp.800-218.pdf)
and
[NASA-STD-8739.8B software safety](https://standards.nasa.gov/standard/nasa/nasa-std-87398)
inform this limited review but do not constitute compliance.

No paid services, cloud compute, app account data, user health,
finance data, or drone hardware were used.
