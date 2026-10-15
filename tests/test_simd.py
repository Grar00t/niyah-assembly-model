#!/usr/bin/env python3
"""Differential test: portable scalar vs CPUID-dispatched AVX2/FMA Assembly.
The Python floating-point oracle only verifies results, it never trains or
runs the shipped neural model. No external packages/network required.
"""
import hashlib
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tests"))
import test_attention as oracle

FLAGS={"AVX2_AUTO":1}
def build(path, macros):
    cmd=["cc","-nostdlib","-static","-no-pie",
         "-Wa,--defsym,EPOCHS=240",
         "-Wa,--defsym,SAMPLE_LEN=0",
         "-Wa,--defsym,EXPORT_WEIGHTS=1"]
    cmd.extend(f"-Wa,--defsym,{name}={val}" for name,val in macros.items())
    cmd.extend(["-o",str(path),str(ROOT/"src/attention_lm.s")])
    subprocess.run(cmd,cwd=ROOT,check=True,capture_output=True,timeout=60)
    obj=subprocess.run(["readelf","-l",str(path)],
                       check=True,capture_output=True).stdout
    assert b"INTERP" not in obj, "unexpected shared runtime"
    obj=subprocess.run(["readelf","-d",str(path)],capture_output=True,check=True).stdout
    assert b"DYNAMIC" not in obj, "unexpected dynamic library"
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(path):
    r=subprocess.run([str(path)],capture_output=True,timeout=60)
    assert r.returncode==0 and not r.stderr, (r.returncode,r.stderr[:120])
    data,params=oracle.parse_params(r.stdout)
    trained=oracle.metric(r.stdout,"final_train_loss_x1000")
    heldout=oracle.metric(r.stdout,"heldout_loss_x1000")
    assert 0 < trained < 350 and 0 < heldout < 500, (trained,heldout)
    error_train=oracle.oracle_loss(oracle.TRAIN,params)*1000
    error_val=oracle.oracle_loss(oracle.VALID,params)*1000
    assert abs(error_train-trained)<3, (error_train,trained)
    assert abs(error_val-heldout)<3, (error_val,heldout)
    return r.stdout,data,params,error_train,error_val

def detect_cpu_features():
    path=Path("/proc/cpuinfo")
    if not path.is_file():return False
    for line in path.read_text().splitlines():
        if line.startswith("flags"):
            items=set(line.split(":",1)[1].split())
            return "avx2" in items and "fma" in items
    return False

def maxdiff(p,q):
    return max(abs(x-y) for a,b in zip(p,q) for x,y in zip(a,b))

def main():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        bins={}
        for name,macros in [("scalar",{}),("auto",{"AVX2_AUTO":1}),
                            ("forced_scalar",{"AVX2_AUTO":1,"FORCE_SCALAR":1})]:
            path=root/name
            build(path,macros)
            bins[name]=path
        cpu=detect_cpu_features()
        if cpu:
            path=root/"avx2"
            build(path,{"AVX2_FAST":1})
            bins["avx2"]=path
        records={name:run(path) for name,path in bins.items()}
        a,b=records["scalar"], records["forced_scalar"]
        assert a[1]==b[1], "fallback differs from scalar kernel weights"
        assert b"kernel_avx2 0\n" in b[0]
        auto=records["auto"]
        if cpu:
            assert b"kernel_avx2 1\n" in auto[0]
            assert auto[1]==records["avx2"][1], "runtime AVX dispatch differs from forced SIMD"
        else:
            assert b"kernel_avx2 0\n" in auto[0]
            assert auto[1]==a[1], "auto CPU fallback differs from scalar"

        # With independently learned Q/K/V, SIMD FP32 reduction changes
        # the nonconvex SGD trajectory. Do not demand bit-identical weights
        # across 240 training epochs: check each path against an independent
        # forward oracle, then compare *observable probabilities and loss*.
        delta=maxdiff(a[2],auto[2])
        loss_train_gap=abs(a[3]-auto[3])
        loss_valid_gap=abs(a[4]-auto[4])
        assert loss_train_gap < 5.0, loss_train_gap
        assert loss_valid_gap < 5.0, loss_valid_gap
        l1=0.0
        for context in (b"the cat ",b"the rat ",b"the mat "):
            dist0=oracle.distribution(context,a[2])
            dist1=oracle.distribution(context,auto[2])
            l1=max(l1,sum(abs(dist0[k]-dist1[k]) for k in dist0))
        assert l1 < 0.02, f"probability changed materially: {l1}"
        # Training persistence: no NaN, no gradient loss, repeatability.
        again=run(bins["auto"])
        assert again[1]==auto[1]
        print("PASS: static self-contained ELF, no interpreter or dynamic runtime")
        print("PASS: true scalar and forced fallback identical trained weights")
        print("PASS: CPUID guarded AVX2/FMA runtime dispatch (reported=%d)"%(cpu))
        print("PASS: independent forward oracle and 240-epoch training (all paths)")
        print("PASS: independently verified loss and output probability equivalence")
        print("PASS: deterministic 240-epoch train repeats")
        print("MEASURE maximum_weight_abs_delta %.9g"%delta)
        print("MEASURE maximum_context_distribution_l1_delta %.9g"%l1)
        print("MEASURE trained_loss_gap_millinats %.9g"%loss_train_gap)
        print("MEASURE heldout_loss_gap_millinats %.9g"%loss_valid_gap)
        print("MEASURE scalar_training_weight_sha256 "+hashlib.sha256(a[1]).hexdigest())
        print("MEASURE optimized_training_weight_sha256 "+hashlib.sha256(auto[1]).hexdigest())
if __name__=="__main__":main()
