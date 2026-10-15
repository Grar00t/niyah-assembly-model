#!/usr/bin/env python3
"""Native Assembly-only training timings, interleaved and repeatable locally.
The Python harness measures subprocess wall times; the model is pure x86-64 asm.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"src"/"attention_lm.s"
def build(exe, mode, epochs):
    c=["cc","-nostdlib","-static","-no-pie",
       f"-Wa,--defsym,EPOCHS={epochs}","-Wa,--defsym,SAMPLE_LEN=0"]
    if mode=="auto":c.append("-Wa,--defsym,AVX2_AUTO=1")
    if mode=="avx2_forced":c.append("-Wa,--defsym,AVX2_FAST=1")
    c+=["-o",str(exe),str(SOURCE)]
    subprocess.run(c,cwd=ROOT,check=True,capture_output=True,timeout=60)
def measure(exe):
    before=time.perf_counter()
    proc=subprocess.run([str(exe)],capture_output=True,timeout=120)
    elapsed=time.perf_counter()-before
    if proc.returncode or proc.stderr:raise RuntimeError((proc.returncode,proc.stderr[:300]))
    return elapsed
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=1200)
    parser.add_argument("--repeats",type=int,default=12)
    a=parser.parse_args()
    assert a.epochs>0 and 3<=a.repeats<=100
    cpuinfo=Path("/proc/cpuinfo").read_text()
    model=next((line.split(":",1)[1].strip() for line in cpuinfo.splitlines()
                if line.startswith("model name")), "unidentified x86-64")
    flags=next((set(line.split(":",1)[1].split()) for line in cpuinfo.splitlines()
                if line.startswith("flags")),set())
    if not {"avx2","fma"}.issubset(flags):
        raise RuntimeError("AVX2/FMA unsupported; only auto fallback may run. Benchmark not applicable")
    modes=["scalar","auto","avx2_forced"]
    with tempfile.TemporaryDirectory() as tmp:
        binaries={mode:Path(tmp)/mode for mode in modes}
        for mode,exe in binaries.items():build(exe,mode,a.epochs)
        for exe in binaries.values():
            for _ in range(2): measure(exe)  # warmup
        times={mode:[] for mode in modes}
        for i in range(a.repeats):
            # Rotate order to reduce drift bias
            order=modes[i%3:]+modes[:i%3]
            for mode in order:times[mode].append(measure(binaries[mode]))
    medians={mode:statistics.median(times[mode]) for mode in modes}
    report={
        "host":"DESKTOP-2P1JAIV (Ubuntu under WSL)",
        "cpu_model":model,
        "epochs":a.epochs,"repeated_trials_per_kernel":a.repeats,
        "median_seconds":{k:round(v,6) for k,v in medians.items()},
        "speedup_auto_vs_scalar":round(medians["scalar"]/medians["auto"],4),
        "speedup_forced_avx2_vs_scalar":round(medians["scalar"]/medians["avx2_forced"],4),
        "all_seconds":{k:[round(n,6) for n in v] for k,v in times.items()},
        "method":"warm twice, rotate invocation order, subprocess run perf_counter medians",
        "limits":"93-byte corpus with 1200 epochs; clock/scheduling may vary; not general LLM throughput"}
    out=ROOT/"evidence"/"simd-benchmark-local.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
