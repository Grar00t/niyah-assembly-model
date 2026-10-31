# Checkpoint and executable review

## Current regression status

The calculator and arithmetic-auditor checks pass. `make verify-all` fails on the current byte-model source. Exact commands, exit codes and logs are in [`evidence/docs-refresh-20261031/build-results.json`](../evidence/docs-refresh-20261031/build-results.json). Historical PASS receipts do not establish a current repository-wide PASS.

## Checkpoint verification-to-use boundary

`tools/checkpoint_attention.py::verified_inference_snapshot` copies source, corpora and weights into a private directory and verifies the copied bytes before compiling there. `tests/test_checkpoint_snapshot.py` mutates disposable originals around that boundary. The intended behavior is isolation from changes after snapshot creation and rejection of a mismatch during capture.

The current context/layout mismatch is documented in [ATTENTION.md](ATTENTION.md). A snapshot identity check cannot repair a wrong layout. The full regression did not reach every later target; no unexecuted test is reported as a fresh PASS. [`evidence/docs-refresh-20261031/verify-all.log`](../evidence/docs-refresh-20261031/verify-all.log)

The manifest is unsigned. A writer who can replace it or modify the private snapshot is outside the protection provided by digest comparison. Weight and manifest replacements are separate operations; interruption can leave a mismatched pair.

## Calculator boundary

The restored calculator uses bounded integer inputs, checked intermediates and standard Euclid GCD reduction. Its independent oracle, parser failures and ELF checks are recorded in [`evidence/docs-refresh-20261031/exactcalc.log`](../evidence/docs-refresh-20261031/exactcalc.log). The arithmetic auditor trusts the executable located beside it. [Exact math](EXACT_MATH.md); [auditor](NATIVE_AUDITOR.md).

## Remaining scope

Assembly buffers are not covered by compiler sanitizers. Tested canaries and selected arithmetic vectors do not establish absence of memory defects. The Qwen report does not verify official weight-file digests. [`tools/qwen/verify/full_gate_g32all_20261031_133521.json`](../tools/qwen/verify/full_gate_g32all_20261031_133521.json)

NIST's Secure Software Development Framework and NASA's Software Assurance and Software Safety Standard remain review references, not compliance claims. Their names, URLs and license status are retained in [SOURCES.md](SOURCES.md).
