#!/usr/bin/env python3
"""Test whether the native Assembly attention probabilities complement n-grams.

No tuning on final test: mix weights are selected using validation rows 0..5,
which had previously been published as an exploratory benchmark.
Final test is pre-registered to TinyStories validation rows 100..107.
The test is reported once. Statistical n-gram code lives only in Python;
the neural probability stream is produced by native x86-64 Assembly.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
import pathlib
import re
import shutil
import struct
import subprocess
import tempfile
import urllib.parse
import urllib.request

ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("tinystories_reference",ROOT/"tools/evaluate_tinystories.py")
source=importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
CTX=source.CTX
LAMBDA_GRID=(0.0,0.05,0.1,0.2,0.4,0.6,0.8,1.0)
FRESH_TEST_OFFSET=100
FRESH_TEST_COUNT=8
FRESH_TEST_SHA256="13140d20d59ed4aebd0bc088a0fdb78fe8292eb51e39230d35ae4b7126ae97c1"

def count_model(train, k=2, alpha=0.5):
    if len(train)<CTX+1:raise ValueError("train too short")
    vocabulary=set(train)
    counts=collections.defaultdict(collections.Counter)
    totals=collections.Counter()
    for i in range(k,len(train)):
        ctx=train[i-k:i] if k else b""
        counts[ctx][train[i]]+=1
        totals[ctx]+=1
    return counts,totals,alpha*len(vocabulary)

def true_probs_ngram(train, valid, k=2, alpha=0.5):
    if not set(valid)<=set(train):raise ValueError("validation OOV: evaluation unsupported")
    counts,totals,denom=count_model(train,k,alpha)
    return [(counts[valid[i-k:i]][valid[i]]+alpha)/(totals[valid[i-k:i]]+denom)
            for i in range(CTX,len(valid))]

def native_true_probs(train,valid,epochs=32):
    if epochs<0 or epochs>100:raise ValueError("epochs out of bounds")
    if not set(valid)<=set(train):raise ValueError("validation OOV: evaluation unsupported")
    with tempfile.TemporaryDirectory(prefix="niyah-prob-export-") as folder:
        root=pathlib.Path(folder)
        (root/"src").mkdir()
        (root/"data").mkdir()
        shutil.copy2(ROOT/"src/attention_lm.s",root/"src/attention_lm.s")
        (root/"data/corpus.txt").write_bytes(train)
        (root/"data/heldout.txt").write_bytes(valid)
        subprocess.run(["cc","-nostdlib","-static","-no-pie",
                        f"-Wa,--defsym,EPOCHS={epochs}",
                        "-Wa,--defsym,SAMPLE_LEN=0",
                        "-Wa,--defsym,AVX2_AUTO=1",
                        "-Wa,--defsym,DUMP_VALID_PROBS=1",
                        "-o","attention_lm","src/attention_lm.s"],cwd=root,
                       capture_output=True,check=True,timeout=45)
        p=subprocess.run([str(root/"attention_lm")],cwd=root,capture_output=True,check=True,timeout=120)
    if p.stderr:raise RuntimeError("Assembly emitted stderr: "+repr(p.stderr[:80]))
    magic=b"VALID_TRUE_PROBS_F32\n"
    if p.stdout.count(magic)!=1:raise ValueError("missing or duplicate probability stream")
    header,_,data=p.stdout.partition(magic)
    expected=4*(len(valid)-CTX)
    if len(data)!=expected:raise ValueError(f"unexpected byte count {len(data)} != {expected}")
    vals=[v[0] for v in struct.iter_unpack("<f",data)]
    if any(not math.isfinite(p) or p<=0 or p>1 for p in vals):
        raise ValueError("non-finite or masked probability")
    label=re.search(rb"^heldout_loss_x1000 (\d+)$",header,re.M)
    if not label:raise ValueError("missing native validation loss")
    native_reported=int(label.group(1))/1000
    recomputed=nll(vals)
    # Native emits rounded int milli-nats, independently recomputed
    # from the actual scored probabilities read via the raw float stream.
    if abs(native_reported-recomputed)>0.001:
        raise AssertionError(("native probability stream disagrees with native NLL",
                              native_reported,recomputed))
    return vals,{"native_reported_nll":native_reported,"independent_nll":round(recomputed,9)}

def nll(ps):
    if not ps:raise ValueError("empty distribution")
    if any(p<=0 or p>1 or not math.isfinite(p) for p in ps):raise ValueError("bad probability")
    return -sum(map(math.log,ps))/len(ps)

def blended(p_ngram,p_neural,lam):
    if len(p_ngram)!=len(p_neural):raise ValueError("wrong counts")
    if not 0<=lam<=1:raise ValueError("weight out of bounds")
    return [(1-lam)*p0+lam*p1 for p0,p1 in zip(p_ngram,p_neural)]

def load_fresh_test():
    q=urllib.parse.urlencode({"dataset":source.DATASET,
                              "config":"default","split":"validation",
                              "offset":FRESH_TEST_OFFSET,"length":FRESH_TEST_COUNT})
    req=urllib.request.Request(source.API+"?"+q,headers={"User-Agent":"niyah-hybrid-honest-test/1.0"})
    with urllib.request.urlopen(req,timeout=40) as response:rows=json.load(response)["rows"]
    if len(rows)!=FRESH_TEST_COUNT or [r["row_idx"] for r in rows]!=list(range(FRESH_TEST_OFFSET,FRESH_TEST_OFFSET+FRESH_TEST_COUNT)):
        raise RuntimeError("fresh test rows mismatch")
    texts=[r["row"]["text"] for r in rows]
    data=("\n\n".join(texts)+"\n").encode().lower()
    h=hashlib.sha256(data).hexdigest()
    if h!=FRESH_TEST_SHA256:raise ValueError(f"FRESH_TEST_REVISION {h}, not scored")
    return data,[hashlib.sha256(t.encode()).hexdigest() for t in texts]

def experiment(epochs):
    train,dev,train_dev_doc_hashes=source.load_documents()
    dev_neural,dev_receipt=native_true_probs(train,dev,epochs)
    dev_ngram=true_probs_ngram(train,dev,2,0.5)
    # Choice is made *solely* on already explored calibration data.
    calibration={str(lam):round(nll(blended(dev_ngram,dev_neural,lam)),9) for lam in LAMBDA_GRID}
    selected=min(LAMBDA_GRID,key=lambda lam:(calibration[str(lam)],lam))
    test,test_docs=load_fresh_test()
    if set(train_dev_doc_hashes)&set(test_docs):
        raise ValueError("duplicate exact story across train/calibration/fresh test")
    if not set(test)<=set(train):raise ValueError("locked fresh test OOV, abort, do not cherry pick")
    test_neural,test_receipt=native_true_probs(train,test,epochs)
    test_ngram=true_probs_ngram(train,test,2,0.5)
    plain_gram=nll(test_ngram)
    mixed=nll(blended(test_ngram,test_neural,selected))
    return {
      "protocol":"2026-10-16-predeclared-unseen-valid-offset-100",
      "data_url":"https://huggingface.co/datasets/roneneldan/TinyStories",
      "reference_paper_url":"https://aclanthology.org/D16-1124/",
      "neural_runtime":"pure x86-64 Assembly, 9616 floats, native masked softmax",
      "statistical_runtime":"CPython n-gram + linear probability interpolation; not part of Assembly binary",
      "epochs":epochs,"lambda_grid":list(LAMBDA_GRID),"lambda_interpretation":"weight of native attention in mixture; 0 = pure trigram",
      "lambda_selected_from_calibration_only":selected,
      "calibration_source":"validation official split rows 0..5 (previously measured)",
      "train_source":"train official split rows 0..11",
      "fresh_test_source":"validation official split rows 100..107",
      "train_sha256":hashlib.sha256(train).hexdigest(),
      "calibration_sha256":hashlib.sha256(dev).hexdigest(),
      "fresh_test_sha256":hashlib.sha256(test).hexdigest(),
      "fresh_test_source_doc_sha256":test_docs,
      "train_doc_count":12,"calibration_doc_count":6,"test_doc_count":8,
      "train_bytes":len(train),"calibration_bytes":len(dev),"fresh_test_bytes":len(test),
      "no_raw_dataset_committed":True,
      "calibration_losses_nats_per_byte":calibration,
      "calibration_native_receipt":dev_receipt,
      "fresh_test_native_receipt":test_receipt,
      "fresh_test_trigram_nll":round(plain_gram,9),
      "fresh_test_native_nll":round(nll(test_neural),9),
      "fresh_test_hybrid_nll":round(mixed,9),
      "fresh_test_hybrid_minus_trigram":round(mixed-plain_gram,9),
      "outperform_trigram_on_fresh_test":mixed<plain_gram,
      "metric":"mean negative log of true next-byte probability, nats/byte, positions 8..N-1",
      "caveats":["No independent hidden benchmark performed by a third party.",
                 "Only 12 training stories; previously observed calibration slice.",
                 "Lambda is a static mixture; not a trained neural gating model.",
                 "Validation records are document-disjoint by split indices; near-duplicates not ruled out.",
                 "Unseen test is a pre-registered new offset and not used to tune this run."]
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--epochs",type=int,default=32)
    p.add_argument("--save",type=pathlib.Path,default=ROOT/"evidence/hybrid-fresh-test-local.json")
    args=p.parse_args()
    result=experiment(args.epochs)
    args.save.parent.mkdir(parents=True,exist_ok=True)
    args.save.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("fresh_test_source_doc_sha256",)},indent=2))
if __name__=="__main__":main()
