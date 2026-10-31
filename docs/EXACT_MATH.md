# Rational calculator

`src/exactcalc.s` implements bounded rational arithmetic using standard Euclid GCD reduction. It was restored from Git history; the source identity is in [`evidence/docs-refresh-20261031/exactcalc-restore.json`](../evidence/docs-refresh-20261031/exactcalc-restore.json). It is separate from the language model.

```sh
make exactcalc test-exactcalc
./build/exactcalc add 1 10 2 10
```

The example returns `3/10`; the executed check is in [`evidence/docs-refresh-20261031/exactcalc.log`](../evidence/docs-refresh-20261031/exactcalc.log).

Arguments are `add|sub|mul|div NUM_A DEN_A NUM_B DEN_B`. Numerators are decimal signed32 values; denominators are positive and at most 2147483647. Arithmetic uses checked signed64 intermediates. The parser rejects plus signs, whitespace, non-ASCII digits, excess arguments and out-of-range input. Input bounds and test vectors are recorded in [`evidence/docs-refresh-20261031/contract-excerpts.json`](../evidence/docs-refresh-20261031/contract-excerpts.json).

Output is a reduced numerator/denominator with positive denominator. Diagnostics go to stderr. The executable uses Linux write/exit syscalls and has no dynamic allocation or filesystem operations.

| Check | Result | Evidence |
| --- | --- | --- |
| Exact outputs vs independent `fractions.Fraction` | 5897 passed | [`evidence/docs-refresh-20261031/exactcalc.log`](../evidence/docs-refresh-20261031/exactcalc.log) |
| Malformed/zero-divisor vectors | 24 rejected with expected codes | [`evidence/docs-refresh-20261031/exactcalc.log`](../evidence/docs-refresh-20261031/exactcalc.log) |
| Static ELF and stack check | PASS | [`evidence/docs-refresh-20261031/exactcalc.log`](../evidence/docs-refresh-20261031/exactcalc.log) |

`tests/test_exactcalc.py` defines parser and arithmetic failure expectations. Broken output pipes retain the process's inherited SIGPIPE behavior. Supported bounded arithmetic does not establish arbitrary precision or language reasoning.

Sources retained: Cornell's Euclid GCD notes; Donald Knuth's *The Art of Computer Programming*; Python `fractions`; John R. Hauser's Berkeley TestFloat. Names, URLs and license status are in [SOURCES.md](SOURCES.md). TestFloat is a testing reference, not a runtime dependency.
