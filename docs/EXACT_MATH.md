# Exact rational arithmetic — native x86-64 Assembly

**Scope:** `src/exactcalc.s`, a bounded exact calculator in the existing
Niyah repository, separate from the educational neural model. It does not
make the language model capable of general reasoning.

## Build and use

Requires Linux x86-64/WSL, GNU assembler (via GCC/`cc`), and GNU Make.
**No MSVC `cl.exe`, libc, libm, Python runtime, network, paid service or GPU.**

~~~sh
make exactcalc
./build/exactcalc add 1 10 2 10       # 3/10 exactly
./build/exactcalc add 1 3 1 6         # 1/2
./build/exactcalc sub 2 3 1 6         # 1/2
./build/exactcalc mul -17 23 23 51    # -1/3
./build/exactcalc div 5 7 -2 9        # -45/14
./build/exactcalc --help
make test-exactcalc
~~~

General form: `exactcalc add|sub|mul|div NUM1 DEN1 NUM2 DEN2`.
Each numerator is decimal signed 32-bit: -2147483648..2147483647.
Each denominator is a *positive* decimal integer: 1..2147483647.
A leading minus is allowed for a numerator. A leading plus, whitespace,
non-ASCII digits, malformed operations, surplus arguments and out-of-range
values are all rejected.

Successful output is one canonical reduced fraction `N/D` plus newline,
with `D>0` and `gcd(|N|,D)=1`. Zero is `0/1`. Diagnostics go to stderr
and do not repeat attacker-controlled text.

## Mathematical contract and bounded correctness

For rational values `n1/d1` and `n2/d2`:

- Addition: `(n1*d2+n2*d1)/(d1*d2)`
- Subtraction: `(n1*d2-n2*d1)/(d1*d2)`
- Multiplication: `(n1*n2)/(d1*d2)`
- Division: `(n1*d2)/(d1*n2)`, provided `n2 != 0`;
  if its denominator is negative, move the sign to numerator.
- Normalize with **Euclid's greatest common divisor**:
  keep replacing `(a,b)` with `(b,a mod b)` until `b=0`.
  Divide numerator and denominator by the resulting nonzero GCD.

The strict parser's small input domain guarantees intermediate signed
64-bit values stay in range:

~~~text
|n1*d2 +/- n2*d1| <= 2 * 2^31 * (2^31 - 1)
                   = 2^63 - 2^32 < 2^63
|n1*n2|            <= 2^62 < 2^63
d1*d2               <= (2^31 - 1)^2 < 2^62
|n1*d2|, |d1*n2|  <= 2^31 * (2^31 - 1) < 2^63
~~~

The routine also checks x86-64 overflow flags after multiplication and
sum/difference (defense-in-depth); the signed division's denominator is
nonzero, and its operands exclude the `INT64_MIN/-1` edge condition.
For supported inputs, exact integer computation preserves rational
values without floating-point rounding.

**Not arbitrary precision**: a numerator or denominator outside the
specified *input* range is rejected even when its value could be represented
by a 64-bit result.

## Failure contract

| Exit | Meaning | stdout |
| ---: | --- | --- |
| 0 | Correct fraction or `--help` | Fraction or help |
| 2 | Wrong argc, operation, malformed or out-of-range data | Empty |
| 3 | Division by rational zero | Empty |
| 4 | Unexpected intermediate overflow | Empty |
| 5 | Write syscall returns an error or zero bytes | May be partial |

A broken stdout pipe may instead terminate the process with SIGPIPE
under the Linux default signal disposition (often status 141 from shells);
the program does not override SIGPIPE or guarantee exit status 5 for it.

Each numeric argument is read for at most ten decimal digits before
rejection; argv is NUL terminated by the OS. No dynamic allocation,
filesystem access, environment variable, shell, network, or actuators.

## Verification and limits

~~~sh
make test-exactcalc
make verify-all
~~~

The test oracle uses independent Python standard-library
`fractions.Fraction` with arbitrary-precision integers only in the
**test harness**. It compares exact output bytes across boundary, seeded
random and invalid-input cases and checks static ELF dependencies, no
undefined runtime symbols, and a non-executable stack.

Local results on DESKTOP-2P1JAIV, Ubuntu/WSL (2026-10-15):

- **5,897** correct calculations, each exactly equal to `Fraction`.
- **24** invalid inputs or zero-divisor cases rejected as specified.
- Native static ELF, no interpreter or dependent shared libraries;
  non-executable GNU_STACK.
- `1/10 + 2/10` returns exactly `3/10` (contrasts with binary floating
  approximation of decimal 0.1+0.2).

These measurements do **not** prove universal absence of defects or
establish downstream benefit to a real person. A user-benefit experiment
would time a real fractional-calculation workflow and count mistakes
against manual or floating-point methods.

## Research and credit

- [Cornell CS Euclidean GCD algorithm and proof](https://www.cs.cornell.edu/courses/cs2800/2016sp/lectures/lec11-gcd.html)
- [Donald Knuth, The Art of Computer Programming Vol. 2](https://cs.stanford.edu/~knuth/taocp.html)
- [Python's independent Fraction oracle](https://docs.python.org/3/library/fractions.html)
- [Berkeley TestFloat by John Hauser](https://www.jhauser.us/arithmetic/TestFloat.html)

This new routine is original code written for Niyah. No source code from
third-party numerical libraries or repositories was copied.

## Dataset, attention, and drone research

The existing attention LM has three trainable Q/K/V matrices already,
but only an 8-byte context and 9,616 learned weights. The
[GSM8K word-problem dataset](https://huggingface.co/datasets/openai/gsm8k)
is interesting for **future evaluation**, yet attaching it now would
not turn this tiny byte predictor into a mathematical reasoner.

A drone control system is safety-critical. This calculator is **not**
connected to a motor, transmitter, sensor, autopilot or flight control
loop. A responsible future milestone would first be an offline
non-actuating simulator with hazard analysis and independent verification,
in the spirit of
[NASA-STD-8739.8B](https://standards.nasa.gov/standard/nasa/nasa-std-87398).
