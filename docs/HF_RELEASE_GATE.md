# Hugging Face release gate — NIYAH Assembly, 2026-10-17

## Current disposition: NOT PUBLISHED TO HUGGING FACE

The existing [GitHub PR #8](https://github.com/Grar00t/niyah-assembly-model/pull/8)
is publicly viewable. This is a legitimate, testable educational project,
but it has NOT yet demonstrated benefit for a real end user. Better
next-byte loss is not a substitute for a product task completed by a person.

The existing connected Hugging Face OAuth session was authenticated as
the author's account with repository **read** scopes, not repository write.
The authorized WSL environment had no configured Hub access token.
The public GitHub repository currently has **no LICENSE** file. Assigning
MIT, Apache-2.0 or another license without the owner's explicit choice
would create a rights claim not supported by the available evidence.

These are independent gates: a clean GitHub CI does not prove Hugging Face
write authority or permission to redistribute source or external data.

## Candidate public description / model card facts

**Name:** NIYAH Assembly — small next-byte ML and exact arithmetic in GNU x86-64.

**Classification:** Educational/research implementation (source code), NOT a
frontier language model, not an Assembly-code interpreter, not an LLM compiler
and not a production model.

**Architecture:** A trainable 9,616-float single-head causal-attention-
inspired byte-level predictor with 8 bytes of context, and a separate
pure Assembly exact-rational calculator. Native training, forward pass,
probability export and inference use static ELF and Linux syscalls. Python
is used for optional external references, tests, checkpoint handling and
research scoring.

**Capabilities actually observed:** Network-free build and regression
suite passed on x86-64 Ubuntu under WSL. A first preregistered evaluation
on official TinyStories validation rows 1200..1223 (24 stories) measured
hybrid NLL 2.069129 versus alpha-tuned two-byte count NLL 2.258089, with
a paired per-story bootstrap descriptive 95% difference interval
[-0.212287, -0.163354]. **Only six target bytes were new to training**;
on those bytes the tuned count baseline won. The hybrid combines Python
count statistics with the **native Assembly** neural probability stream;
it is not a fully native hybrid generator.

**Appropriate use:** teaching low-level neural-network implementation,
independent correctness tests, reproducible training receipts and careful
evaluation. It must not be advertised as helpful for user conversation,
binary security auditing or human language comprehension.

**Known limitations:** Single attention head and 8-byte context;
tiny training sample; one seed; fixed, static mixture;
model has no meaningful semantic task evaluation; no explicit code license;
no directly transferable all-byte v2 checkpoint saved for publication.
Linux x86-64 and a toolchain are needed to reproduce training.
No large Assembly-code dataset or model weights are redistributed.
TinyStories source carries CDLA-Sharing-1.0 metadata and is fetched for
local evaluation only.

**Research attribution:** Wang et al. ASMA-Tune (2025);
SuperCoder (arXiv:2505.11480, distinguish v1 from v3);
CompilerEval/LaaC (2025); Neubig and Dyer (2016);
Eldan and Li TinyStories (2023).
Full protocol and honest results:
docs/PREREGISTERED_EVAL.md and docs/PREREGISTERED_RESULTS_20261017.md.

## Release checklist (not automatic)

- [x] Source is public on GitHub PR, with a frozen protocol and a real test receipt
- [x] Local GNU x86-64 build plus full native/offline regression passed
- [x] Non-blind prior experiment separated from newly preregistered test
- [x] External licensed text NOT added to the project
- [ ] The owner chooses and publishes a legally appropriate license for source
- [ ] The owner grants a Hugging Face authorization with repository write permission
- [ ] Choose the correct artifact class: code example/demo Space, model repository
      containing actual supported checkpoints, or dataset repository with legal rights
- [ ] If a Space: test a safe fixed demo without arbitrary user-supplied machine code
- [ ] Publish a complete repository card including negative OOV findings, runtime
      constraints, data provenance, reproduction commands and all limitations
- [ ] Independently verify that the remote Hugging Face repository exists and
      matches its tagged GitHub source and checksums
- [ ] Have at least one real external learner run the task and record whether
      they can reproduce the result; do not invent human-impact metrics

**Next benefit experiment:** Ask a new learner who has not seen the code to
clone and run make verify-all and explain the difference between masked and
all-byte OOV NLL. Record steps completed, error count and time-to-reproduce.
That measures an educational task rather than equating a lower NLL with
human benefit.
