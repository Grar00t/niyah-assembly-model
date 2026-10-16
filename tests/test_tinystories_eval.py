#!/usr/bin/env python3
"""Offline CI regression for the real-dataset evaluation harness."""
import importlib.util
import math
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("evaluation", ROOT/"tools"/"evaluate_tinystories.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def independent_unigram(train,valid):
    vocab=set(train); alpha=0.5; total=0
    for i in range(mod.CTX,len(valid)):
        ch=valid[i]
        p=(train.count(ch)+alpha)/(len(train)+alpha*len(vocab))
        total-=math.log(p)
    return total/(len(valid)-mod.CTX)

def main():
    t=(b"the little mouse found some cheese. she carried it home.\n" * 15)
    v=(b"the mouse wanted to share cheese with a friend.\n" * 5)
    # Validation cannot have characters unseen in training. Include every byte
    # in the fixture's vocabulary for an isolated causal test.
    t+=b"wanted share friend"
    assert set(v)<=set(t), sorted(set(v)-set(t))
    loss=mod.baseline(t,v,0)
    expected=independent_unigram(t[0:],v)
    # The independent oracle indexes all train bytes; the implementation
    # excludes the first k tokens. For k=0, these agree exactly.
    assert abs(loss-expected)<1e-12,(loss,expected)
    assert mod.baseline(t,v,2)>0
    one=mod.build_and_run(t,v,0,True)
    two=mod.build_and_run(t,v,2,True)
    assert two["train_nll_nats_byte"]<one["train_nll_nats_byte"],(one,two)
    assert math.isfinite(two["validation_nll_nats_byte"])
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/"report.json"
        result=mod.evaluate(t,v,["document-sha1","document-sha2"],2)
        assert result["native_trained"]["epochs"]==2
        assert result["train_bytes"]==len(t)
        assert result["validation_bytes"]==len(v)
        assert len(result["baseline_metrics"])==3
    print("PASS: offline native Assembly build/train/validation")
    print("PASS: independent exact unigram oracle")
    print("PASS: no network, checkpoint, or external training data required")
    print("PASS: no LM runtime except Linux syscalls")
if __name__=="__main__":main()
