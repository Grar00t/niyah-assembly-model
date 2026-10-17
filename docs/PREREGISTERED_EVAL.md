# NIYAH preregistered story-level byte evaluation protocol

**Protocol:** NIYAH-2026-10-17-preregistered-story-OOV-v1
**Status:** Pre-register source code and fixed test row IDs on GitHub BEFORE first fetch of these test rows.
**Purpose:** Evaluate whether a tiny native Assembly model complements a count-based model without picking convenient data or scoring artificial cross-story context.

## Beneficiary and benefit hypothesis

Students of low-level ML, reproducibility researchers, and Assembly maintainers can inspect an experiment with a hand-written x86-64 model, an independent count baseline, and bounded data. Proposed benefit: a repeatable example of comparing native arithmetic to a reference implementation without buying GPUs or using a paid service. **No independent user study or real-world societal benefit has been measured.**

## Fixed choices (no changes after viewing the test)

| Field | Preregistered selection |
| --- | --- |
| Dataset | roneneldan/TinyStories, config default, official validation split; licensed CDLA-Sharing-1.0 |
| Train | train rows 0..11, previously published sample |
| Calibration | validation rows 0..5, previously measured, used ONLY for model selection |
| **New test** | **validation rows 1200..1223 (24 stories)**, not queried by this project before registration |
| Preprocessing | Encode each story as UTF-8, lowercase ASCII bytes, separate records with two LF bytes, append one trailing LF |
| Neural implementation | Current native GNU x86-64 Assembly, one head, 8-byte context, 9616 float32 params, all 256 byte IDs enabled |
| Training | 32 epochs, current fixed seed 2463534242, safe runtime AVX2 dispatch |
| Baseline candidates | orders {0,1,2}; positive add-alpha from {0.001,0.01,0.05,0.1,0.25,0.5,1} over 256 IDs |
| Mixture candidates | Native mixing weight {0,0.05,0.1,0.2,0.4,0.6,0.8,1} |
| Selection | Choose n-gram order/alpha and then mix weight on calibration **only**; deterministic minimal NLL, fixed ties |
| Test scoring | Exclude first 8 target bytes of EACH story plus all artificial separator bytes |
| Main metric | Mean true next-byte NLL (nats/byte), reported for native, best calibrated count model, hybrid |
| OOV targets | Test byte IDs absent from training; report their count and separate NLL, plus in-vocabulary NLL |
| Uncertainty | 2000 paired bootstrap resamples of the **24 stories**; seed 20261017; descriptive 95% interval for hybrid-minus-count NLL |
| Bound | Maximum 300,000 processed test bytes, refuse duplicate source documents, unexpected row IDs, invalid probability streams |
| Integrity | Frozen train/calibration SHA-256 must match. New test text hashes recorded after first permitted post-registration retrieval; no raw stories committed |

All comparison methods use the **same** context and native probability stream. Native float32 may underflow for rare OOV targets; fail the evaluation rather than discarding such positions or substituting other data.

A negative test-set NLL difference is an improvement **on those measurements**, not automatically a useful text generator. A paired descriptive 95% interval entirely below zero provides stronger evidence for the fixed sample, but does not establish generalization to unrelated domains.

## Reproduction order

1. Commit and push this protocol and its test source to GitHub. Record the registration commit ID.
2. After the commit is visible, run the offline regression gate and then the official benchmark exactly once. Do not change frozen source selection because of the result.
3. Publish the outcome, including failures and OOV-only scores, plus complete SHA-256, command line, machine architecture, and method limitations.

~~~sh
make test-preregistered-eval
make verify-all
python3 tools/preregistered_eval.py --save evidence/preregistered-oov-story-local-20261017.json
~~~

The external run downloads only bounded official story text in memory. Outputs are hashes and aggregated loss numbers, not the original licensed stories. The native model trains and exports true target probabilities in Assembly; the selected n-gram and blending implementation runs in Python solely for evaluation. This is NOT an Assembly-source understanding LLM, a compiler, or a chatbot.

## Research decisions and attribution

- [ASMA-Tune, arXiv:2503.11617 (submitted March 2025)](https://arxiv.org/abs/2503.11617): a specialised Assembly encoder, alignment projector and large decoder help analyse Assembly source. The paper uses a multi-GPU, ~13B-parameter architecture. This is a different research question and hardware scale; do not claim the NIYAH 9616-weight byte LM inherits those abilities.
- [SuperCoder, arXiv:2505.11480](https://arxiv.org/abs/2505.11480): the initial 2025 paper version under the title Improving Assembly Code Performance... reported 1.47x average speedup over gcc -O3 **with** compiler-generated reference Assembly. Its no-O3-ablation reported zero pass rate for the PPO model. The revised January 2026 v3 now describes SuperCoder and reports 1.46x in its abstract. These are study-specific versioned results, not NIYAH speedups.
- [CompilerEval / LLM-as-a-Compiler, arXiv:2511.04132](https://arxiv.org/abs/2511.04132): evaluates compilation and semantic correctness of 20 C kernels. The design principle transferred here is independent correctness checks, NOT replacement of a compiler.
- [Neubig and Dyer, EMNLP 2016](https://aclanthology.org/D16-1124/): the rationale for comparing count and neural probability streams. NIYAH uses calibration-selected FIXED probability interpolation, not a context-adaptive trained gate.
- [Eldan and Li, TinyStories (2023)](https://arxiv.org/abs/2305.07759): the tiny-word-vocabulary dataset motivation. Original licensed text is not redistributed in project commits.

## Release and ethical limits

No verified human task benefit, semantic understanding, or production reliability is established. This public source repository has **no explicit LICENSE file** as of this protocol: a public URL alone grants no reuse license. Do not add a permissive license on the author's behalf or copy third-party dataset content. A Hugging Face release requires an explicit project license choice by the owner, an honest model/research card, and authenticated repository-write access; absent those, publication there is blocked even if GitHub CI succeeds.
