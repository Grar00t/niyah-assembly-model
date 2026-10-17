# NIYAH: first preregistered story-level OOV result (2026-10-17)

## Evidence identity and order of operations

The evaluation protocol, row IDs, calibration grids, target selection,
code and independent offline regression were committed to GitHub first:

- Preregistration commit: f7540b990d7eee0888bd9cb3e6208d80270533b9
- GitHub: https://github.com/Grar00t/niyah-assembly-model/commit/f7540b990d7eee0888bd9cb3e6208d80270533b9
- Source protocol: docs/PREREGISTERED_EVAL.md in that commit
- Frozen evaluation code: tools/preregistered_eval.py in that commit
- Branch/PR: https://github.com/Grar00t/niyah-assembly-model/pull/8
- Verified remote push before first test retrieval; WSL first run
  from 2026-10-17T02:23:26Z to 02:23:42Z; exit 0.
- First-run JSON: evidence/preregistered-oov-story-local-20261017.json
- First-run JSON SHA-256:
  24affc9afe1de33f889ac72dd4554bc34b891dfa245f78a065d11bb992891670

These new test rows were not retrieved by this project until after the
above commit was confirmed on the remote. This makes the test *prospectively
preregistered relative to this development*, NOT a claim that the dataset
was globally secret or that no one has ever investigated similar rows.

## Data and scoring contract

Data: official roneneldan/TinyStories, license metadata CDLA-Sharing-1.0.
Training and calibration were already previously inspected:
train rows 0..11 (8575 processed bytes) and validation rows 0..5
(4212 processed bytes). Test rows were fixed at validation 1200..1223.

Test source: 24 stories, 17044 processed bytes; 16805 within-story
next-byte targets after dropping the first eight bytes of each story
and inter-story separators. This prevents concatenation artifacts from
entering the scored positions. The source test bytes were hashed SHA-256:

3b6a01db09383c69184980dc54f75e11b3e483ef391675f2d674a3c9bb5ad710

Precommitted training: 32 native Assembly SGD epochs, fixed seed.
Count candidate orders 0/1/2 and alpha grid were selected on calibration
only. Selected: **order 2**, **alpha 0.01**, **lambda 0.4** for the
fixed (0.6 count / 0.4 native) probability interpolation.

The neural model is 9616 float32 parameters and eight bytes of context;
its forward, gradient and training steps run in GNU x86-64 Assembly.
The count model, mixture, and statistical summaries are in Python.
No licensed text, extracted model weights or user account secrets
were committed.

## Observed results

| Test next-byte NLL (nats/byte; lower is better) | 256-ID tuned count | Native Assembly | Fixed mixture |
| --- | ---: | ---: | ---: |
| All 16805 within-story targets | 2.258089451 | 2.452823074 | **2.069128925** |
| 16799 targets already in train alphabet | 2.256635350 | 2.449752659 | **2.067429561** |
| Only **6 genuine OOV target bytes** | **6.329330232** | 11.049474663 | 6.827065217 |

Aggregate improvement of the mixture versus the **calibrated** count
baseline is 0.188960526 nats/byte, **8.36816%** on this fixed sample.
The mixture was better than count on **24 of 24 individual stories**.

Deterministic paired whole-story bootstrap, 2000 replicates and seed
20261017, for (hybrid NLL minus tuned count NLL): the descriptive
95-percentile interval was [-0.212287025, -0.163354378] nats/byte.
That excludes zero within the chosen resampling procedure. It is not
a formal independent replication or an estimate of real-world
user-task quality.

The OOV target IDs absent from train were 45 (hyphen) and 58 (colon):
**only 6 OOV target positions** among 16805. On these, the count baseline
is better than the hybrid. It is not fair to claim the OOV problem is
solved based on the aggregate positive result.

An independent post-run aggregation read all 24 per-story score sums,
recomputed all three mean NLLs and target counts, verified the locked
training and source hashes, and confirmed the preregistration and
first-run receipt matched. The native stream also independently checks
its reported NLL against its actual binary probabilities before use.

## Execution and reproduction

System: authorized Ubuntu under Windows Subsystem for Linux, x86-64,
using the native GNU assembler, link tools and static ELF.
The local full gate ran before the remote preregistration:
make verify-all passed including scalar/SIMD differential checks,
independent gradient and arithmetic references, snapshot safety,
offline OOV and story scoring negative tests.

~~~sh
make test-preregistered-eval
make verify-all
python3 tools/preregistered_eval.py --save evidence/preregistered-oov-story-local-20261017.json
sha256sum evidence/preregistered-oov-story-local-20261017.json
~~~

The runner checks source identifiers and frozen training/calibration
hashes, then fetches the official test rows; the first-run SHA-256
can be compared to the published result. The runner will *not*
substitute different test rows or tune hyperparameters on the test.

## Limitations and decision

**Observed value:** a small educational Assembly ML implementation
can produce correct native probability streams and a reproducible
method for comparing hybrid versus count-based next-byte predictions.
This is a verified **methodological/engineering outcome**.

**Human benefit: NOT MEASURED.** There is no assessed real student,
software-maintenance, translation, document-writing, accessibility
or language-comprehension task; better NLL is not equivalent to those.
The model is not an Assembly-code understanding LLM or a compiler,
and its OOV performance remains worse than the tuned count baseline.
Only 12 short training stories, a single seed and 24 test stories
were used, and the source model-selection slice was previously explored.

Do not describe the result as general SOTA or borrow speedup figures
from the unrelated SuperCoder paper. Do not merge or deploy to users
solely from this preliminary positive score.

**Release gate:** The GitHub source PR is publicly reviewable.
Hugging Face publication is **BLOCKED** because the project has no
declared code license, the connected Hub OAuth token has read-only
repository access, and the authorized WSL environment has no token.
The project owner must choose a license and grant write access before
publishing. A model card must plainly state this experiment is
educational and that no transferable full-byte v2 checkpoint exists.

Research/methods: docs/PREREGISTERED_EVAL.md, docs/FULLBYTE_OOV.md;
official source https://huggingface.co/datasets/roneneldan/TinyStories;
hybrid count/neural precedent https://aclanthology.org/D16-1124/ .
