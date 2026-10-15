#!/usr/bin/env python3
"""Numerical central differences for learned Q/K/V against ONE native SGD step."""
import math
import subprocess
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tests"))
import test_attention as oracle

LEARNING_RATE=0.025
EPSILON=0.0075
def build(path,epochs,step=False):
    flags=[f"-Wa,--defsym,EPOCHS={epochs}",
           "-Wa,--defsym,SAMPLE_LEN=0",
           "-Wa,--defsym,EXPORT_WEIGHTS=1"]
    if step:flags.append("-Wa,--defsym,ONE_STEP=1")
    command=["cc","-nostdlib","-static","-no-pie",*flags,
             "-o",str(path),str(ROOT/"src/attention_lm.s")]
    subprocess.run(command,cwd=ROOT,check=True,capture_output=True,timeout=60)
def run(path):
    p=subprocess.run([str(path)],check=True,capture_output=True,timeout=60)
    return oracle.parse_params(p.stdout)[1]

def loss(context,target,arrays):
    p=oracle.distribution(context,arrays)
    return -math.log(max(1e-30,p[target]))

def finite_difference(context,target,arrays,idx,j):
    perturbed=list(arrays)
    changed=list(arrays[idx])
    perturbed[idx]=changed
    value=changed[j]
    changed[j]=value+EPSILON
    positive=loss(context,target,perturbed)
    changed[j]=value-EPSILON
    negative=loss(context,target,perturbed)
    return (positive-negative)/(2*EPSILON)

def main():
    context=oracle.TRAIN[:oracle.T]
    target=oracle.TRAIN[oracle.T]
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/"train"
        build(path,epochs=0)
        initial=run(path)
        build(path,epochs=1,step=True)
        updated=run(path)
    assert len(initial)==9 and len(updated)==9
    for idx,name in [(6,"WQ"),(7,"WK"),(8,"WV")]:
        candidates=[j for j in range(256) if j//16!=j%16]
        biggest=max(candidates,key=lambda j:abs(initial[idx][j]-updated[idx][j]))
        delta=initial[idx][biggest]-updated[idx][biggest]
        native_gradient=delta/LEARNING_RATE
        reference_gradient=finite_difference(context,target,initial,idx,biggest)
        relative_error=abs(native_gradient-reference_gradient)/max(
            1e-12,abs(native_gradient),abs(reference_gradient))
        print(name,"selected_offdiag",biggest,
              "native_gradient",native_gradient,
              "reference_gradient",reference_gradient,
              "relative_error",relative_error,
              "native_step_delta",delta)
        assert abs(delta)>1e-10, f"{name} stayed frozen"
        assert relative_error < 0.08, f"{name} wrong gradient: {relative_error}"
    # One SGD step must improve the loss at the same training example.
    before=loss(context,target,initial)
    after=loss(context,target,updated)
    assert after<before, (before,after)
    print("PASS: independently checked WQ/WK/WV one-step SGD gradients")
    print("LOSS one-step before",before,"after",after)
if __name__=="__main__":main()
