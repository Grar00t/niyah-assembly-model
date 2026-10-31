# Arithmetic candidate auditor

`src/nexus_audit.s` runs the sibling `exactcalc` executable using fork, pipes and exec, then compares its canonical rational output with a supplied candidate. It resolves the sibling through `/proc/self/exe`; it does not search PATH or invoke a shell.

```sh
make nexus-audit test-nexus-audit
```

The calculator and auditor checks completed with process exit 0. The auditor reports 7 tests passed. [`evidence/docs-refresh-20261031/exactcalc.log`](../evidence/docs-refresh-20261031/exactcalc.log)

Arguments are the calculator operation, its operands, then the candidate. The caller must supply the arithmetic expression. A decimal string, an unreduced fraction or prose does not equal the canonical fraction. [Calculator domain](EXACT_MATH.md).

The auditor requires successful child exit and bounded output framing. It handles partial I/O, interruptions and broken stdout. Both executables and their directory must be trusted; a fixed executable path is not an authenticity check. It does not infer expressions from conversation or change model weights.

`tests/test_nexus_audit.py` covers an independent Fraction oracle, malformed candidates and child output, missing sibling, empty PATH, paths with spaces, output failures and static ELF checks. The executed log is [`evidence/docs-refresh-20261031/exactcalc.log`](../evidence/docs-refresh-20261031/exactcalc.log).
