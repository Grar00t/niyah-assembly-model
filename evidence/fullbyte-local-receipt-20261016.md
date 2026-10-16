# Local verification receipt: NIYAH full-byte OOV (2026-10-16)

- Environment: Ubuntu under Windows Subsystem for Linux, x86-64
- Parent main commit: fb57c8355833fdb956e88e3588ccdd6c5448fed8
- Branch: feat/fullbyte-oov-eval-20261016
- Local full-gate command: make verify-all
- Observed result: exit 0; start 2026-10-16T21:04:46Z; finish 2026-10-16T21:05:05Z
- Native OOV regression: make test-fullbyte-oov
- Optional external benchmark: python3 tools/oov_revisit.py --epochs 32
- Dataset: locked TinyStories source hashes, no raw third-party text committed
- Previously inspected OOV evaluation slice, NOT a blind generalization test
- Model scope: one attention head, 9616 float parameters, context 8 bytes
- Count baseline and mixture: Python evaluator, not native Assembly generation
- User benefit: NOT MEASURED

## SHA-256 input file hashes

~~~text
9f321e7fa7e4c7f89cae3585a32084cbd2f85a2a6d5b709ff871273d39f259b6  src/attention_lm.s
d3330eebc661265fb1c082040b592c0842c23f98c973ea15cb3d2c36697b349d  Makefile
eb0ed92b01fcfbed9e5476f90a0adaa83d74ad107e00f641fd06bde487aa1155  tools/hybrid_eval.py
4d4c2ebabda2e1d99fd2b54f10904c4a0e4c42dd205e75cd5be707b5297e7a12  tools/oov_revisit.py
295b633cfa25c6fd69e3d4c21bbbd35d186f2cea0ba3fe3793aa343c986ee717  tests/test_fullbyte_oov.py
e6d2679cc960fcebf5903d03a956f371f886bb41e0fda672f67a3c0244a33f50  evidence/fullbyte-oov-revisit-local.json
f7072621cd4a4852d0c1244cba84c8269a83cc83d9fc6470a9954951fe1cad17  evidence/fullbyte-verify-all-local.txt
~~~

## Evidence

- evidence/fullbyte-verify-all-local.txt - verbatim make verify-all output; native generated sample carries a trailing ASCII space on line 111 (preserved, not source whitespace)
- evidence/fullbyte-oov-revisit-local.json - neural losses, calibrated grids and OOV-only results
- docs/FULLBYTE_OOV.md - implementation details and evidence boundaries

## Research sources

- https://huggingface.co/datasets/roneneldan/TinyStories
- https://aclanthology.org/D16-1124/
