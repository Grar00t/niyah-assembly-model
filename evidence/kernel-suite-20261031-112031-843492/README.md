# Integer GEMM and dot verification suite

Run from this directory with GNU tools and Python:

```sh
python3 run_suite.py --repo /absolute/path/to/niyah-assembly-model --out /absolute/path/to/new-results --opt O2 --full
```

The compiler option above is recorded in [`evidence/kernel-suite-20261031-112031-843492/results-o2-full/metadata.json`](results-o2-full/metadata.json). The output directory must not already exist. `--verify-only` skips timing.

The runner extracts the Assembly literals from `tools/qwen/qwen_native_layer0.py` without importing that module, then links `src/niyah_i8_gemm.s`. It uses an independent integer oracle, scalar floating reference, input immutability checks and output canaries. The integer kernel is called serially because it shares scratch storage.

| Recorded run | Result | Evidence |
| --- | --- | --- |
| Standard | 990 cases PASS | [`evidence/kernel-suite-20261031-112031-843492/results-o2/verification.log`](results-o2/verification.log) |
| Complete shapes | 1002 cases PASS | [`evidence/kernel-suite-20261031-112031-843492/results-o2-full/verification.log`](results-o2-full/verification.log) |

All timings remain in [`evidence/kernel-suite-20261031-112031-843492/results-o2/benchmark.csv`](results-o2/benchmark.csv) and [`evidence/kernel-suite-20261031-112031-843492/results-o2-full/benchmark.csv`](results-o2-full/benchmark.csv). Metadata records compiler flags and source hashes. These local results supersede the old statement that the complete suite had not been executed.

Do not add global vector-target flags: capability dispatch must execute on the baseline ISA. The suite does not test concurrent calls, arbitrary pointers, every read boundary, model quality or g32 quantization. Resident-buffer timings include call overhead.
