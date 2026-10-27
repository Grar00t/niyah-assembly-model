#!/usr/bin/env python3
"""Warm, interleaved, pinned-CPU local measurements. No inference/training in Python."""
import hashlib,json,os,pathlib,statistics,struct,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
executions=[]
def run(cmd,**kw):
    q=subprocess.run(cmd,cwd=ROOT,check=True,capture_output=True,timeout=180,**kw)
    executions.append({"command":[str(x) for x in cmd],"exit_code":q.returncode,"stdout_sha256":hashlib.sha256(q.stdout).hexdigest()})
    return q
def quant(xs,q):return sorted(xs)[min(len(xs)-1,int((len(xs)-1)*q))]
def stats(xs):
    med=statistics.median(xs)
    return {"n":len(xs),"median_seconds":med,"min_seconds":min(xs),"max_seconds":max(xs),"p10_seconds":quant(xs,.1),"p90_seconds":quant(xs,.9),"mad_seconds":statistics.median(abs(v-med) for v in xs)}
def main():
    cores=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{cores[0]})
    exe=ROOT/"build/tile_probe"
    source_names=["src/tile_probe.s","src/tile_kernels.s","src/attention_lm.s","tools/bench_native_tiles.py"]
    source_before={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in source_names}
    kernel_before=hashlib.sha256(exe.read_bytes()).hexdigest()
    f=struct.unpack("<8Q",run([str(exe),"features"]).stdout)
    assert f[1]&1,"AVX2 unsupported"
    n=100000
    payload=struct.pack("<512f",*([float(i%17-8) for i in range(256)]*2))
    paths={"scalar":[str(exe),"scalar",str(n)],"avx2_tiled":[str(exe),"avx2",str(n)]}
    hashes={}
    def measure(cmd,inp):
        t=time.perf_counter();p=run(cmd,input=inp);s=time.perf_counter()-t
        return s,hashlib.sha256(p.stdout).hexdigest()
    for name,cmd in paths.items():
        for _ in range(2):_,hashes[name]=measure(cmd,payload)
    assert len(set(hashes.values()))==1,"benchmark outputs differ"
    samples={k:[] for k in paths}
    names=list(paths)
    for i in range(11):
        order=names if i%2==0 else names[::-1]
        for k in order:
            elapsed,h=measure(paths[k],payload);assert h==hashes[k];samples[k].append(elapsed)
    gemm_stats={k:stats(v) for k,v in samples.items()}
    # Compare the same AVX2 model with and without tiled Q/K/V, not scalar vs SIMD.
    bins={}
    for name,extra in [("dot16_avx2",[]),("tiled_projection_avx2",["-Wa,--defsym,TILED_PROJECTIONS=1"])]:
        dst=ROOT/"build"/("bench_"+name);bins[name]=dst
        cmd=["cc","-nostdlib","-static","-no-pie","-Wa,--defsym,AVX2_AUTO=1","-Wa,--defsym,EPOCHS=1200","-Wa,--defsym,SAMPLE_LEN=0"]+extra+["-o",str(dst),"src/attention_lm.s"]
        run(cmd)
    model_hashes={}
    for name,dst in bins.items():
        for _ in range(2):_,model_hashes[name]=measure([str(dst)],b"")
    assert len(set(model_hashes.values()))==1,"model benchmark observable output differs"
    ms={k:[] for k in bins};names=list(bins)
    for i in range(11):
        for k in (names if i%2==0 else names[::-1]):
            elapsed,h=measure([str(bins[k])],b"");assert h==model_hashes[k];ms[k].append(elapsed)
    model_stats={k:stats(v) for k,v in ms.items()}
    cpu=next(v.split(":",1)[1].strip() for v in pathlib.Path("/proc/cpuinfo").read_text().splitlines() if v.startswith("model name"))
    report={"cpu":cpu,"pinned_cpu":cores[0],"method":"2 warmups; 11 alternating-order subprocess wall-time samples per path; identical output checked each run",
      "gemm":{"shape":"FP32 16x16 * 16x16","native_repetitions_per_sample":n,"stats":gemm_stats,"all_seconds":samples,"speedup_scalar_over_avx2_tiled":gemm_stats["scalar"]["median_seconds"]/gemm_stats["avx2_tiled"]["median_seconds"],
       "avx2_gflop_per_second":(n*2*16**3)/gemm_stats["avx2_tiled"]["median_seconds"]/1e9},
      "model":{"epochs":1200,"training_bytes":(ROOT/"data/corpus.txt").stat().st_size,"stats":model_stats,"all_seconds":ms,
       "speedup_existing_avx2_over_tiled_projection":model_stats["dot16_avx2"]["median_seconds"]/model_stats["tiled_projection_avx2"]["median_seconds"],
       "output_sha256":model_hashes},
      "limitations":"Fixed tiny matrices and 93-byte training fixture; WSL scheduling/process overhead included; no AMX/AVX512 performance measured; not general LLM throughput."}
    source_after={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in source_names}
    assert source_before==source_after
    assert kernel_before==hashlib.sha256(exe.read_bytes()).hexdigest()
    report["source_sha256"]=source_after
    report["binary_sha256"]={"tile_probe":kernel_before,**{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in bins.items()}}
    report["all_process_exit_codes_observed"]=executions
    report["source_and_kernel_hashes_unchanged_during_measurement"]=True
    out=ROOT/"evidence/nexus-native-benchmark.json";out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in {"method"}},indent=2))
if __name__=="__main__":main()
