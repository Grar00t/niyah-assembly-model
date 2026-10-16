#!/usr/bin/env python3
"""Network-free, native x86-64 probability-stream + mixing regression."""
import collections
import importlib.util
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location("hybrid",ROOT/"tools/hybrid_eval.py")
m=importlib.util.module_from_spec(sp)
sp.loader.exec_module(m)

def independent_trigram(train,valid):
    occurrences=collections.Counter(zip(train[:-2],train[1:-1],train[2:]))
    histories=collections.Counter(zip(train[:-2],train[1:-1]))
    vocab_count=len(set(train))
    return [
        (occurrences.get((valid[i-2],valid[i-1],valid[i]),0)+0.5)/
        (histories.get((valid[i-2],valid[i-1]),0)+0.5*vocab_count)
        for i in range(8,len(valid))
    ]

def main():
    train=b"the little mouse found cheese. she carried it home.\n"*16+b"wanted share friend\n"
    valid=b"the mouse wanted to share cheese with a friend.\n"*5
    assert set(valid)<=set(train)
    a=m.true_probs_ngram(train,valid)
    b=independent_trigram(train,valid)
    assert all(abs(x-y)<1e-14 for x,y in zip(a,b)), "trigram oracle mismatch"
    assert abs(m.nll(a)-m.source.baseline(train,valid,2))<1e-11
    neural,record=m.native_true_probs(train,valid,2)
    assert len(neural)==len(valid)-m.CTX
    assert abs(m.nll(neural)-record["native_reported_nll"])<=0.001
    assert abs(m.nll(m.blended(a,neural,0))-m.nll(a))<1e-12
    assert abs(m.nll(m.blended(a,neural,1))-m.nll(neural))<1e-12
    mix=m.blended(a,neural,0.4)
    assert all(0<p<=1 for p in mix)
    assert all(abs(p-(0.6*x+0.4*y))<1e-12 for p,x,y in zip(mix,a,neural))
    try:m.true_probs_ngram(train,valid+b"\xff")
    except ValueError as ex:assert "OOV" in str(ex)
    else:raise AssertionError("OOV should fail closed")
    try:m.blended(a,neural,-0.1)
    except ValueError:pass
    else:raise AssertionError("bad weight should fail closed")
    print("PASS: independent trigram oracle, matching CTX windows")
    print("PASS: native Assembly probability bytes reproduce native NLL")
    print("PASS: mixing weights 0 and 1 exactly recover components")
    print("PASS: nonnegative normalization-domain and fail-closed OOV")
    print("PASS: external dataset and network not required")
if __name__=="__main__":main()
