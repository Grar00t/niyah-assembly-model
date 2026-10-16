#!/usr/bin/env python3
"""TinyStories external, heldout-only benchmark for NIYAH pure-Assembly byte LM.

Never commits any dataset bytes or pulls a model. The byte-level model
receives ASCII-lowercased train/validation corpora in an isolated build directory.
"""
import argparse, collections, hashlib, json, math, pathlib, re, shutil
import subprocess, tempfile, urllib.parse, urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]
DATASET="roneneldan/TinyStories"
API="https://datasets-server.huggingface.co/rows"
CTX=8
TRAIN_N=12
VALID_N=6
ALPHA=0.5
# Immutable corpus contract from the initial checked local benchmark.
EXPECTED_TRAIN_SHA256="d284702c7392bedc3373d0a4f0400870d563265c41d479c6a7d9c57904f3dca4"
EXPECTED_VALID_SHA256="ad1d534077e9e29b21c9e4bb912afa05f309bf5d4cedafcdb77c3b16b9660660"

def request(split,count):
    query=urllib.parse.urlencode({"dataset":DATASET,"config":"default","split":split,"offset":0,"length":count})
    req=urllib.request.Request(API+"?"+query,headers={"User-Agent":"niyah-assembly-model-benchmark/1.0"})
    with urllib.request.urlopen(req,timeout=45) as response:
        content=json.load(response)
    if len(content["rows"])!=count:raise RuntimeError(f"incomplete {split} rows")
    return content["rows"]

def load_documents():
    train=request("train",TRAIN_N)
    valid=request("validation",VALID_N)
    assert [x["row_idx"] for x in train]==list(range(TRAIN_N))
    assert [x["row_idx"] for x in valid]==list(range(VALID_N))
    td=[x["row"]["text"] for x in train]
    vd=[x["row"]["text"] for x in valid]
    hs=[hashlib.sha256(x.encode()).hexdigest() for x in td+vd]
    if len(set(hs))!=len(hs):raise ValueError("duplicated exact story across splits")
    # Explicit preprocessing is the SAME for all methods and both splits.
    # The toy 256-byte model has no out-of-vocabulary fallback.
    train_bytes=("\n\n".join(td)+"\n").encode().lower()
    valid_bytes=("\n\n".join(vd)+"\n").encode().lower()
    if hashlib.sha256(train_bytes).hexdigest()!=EXPECTED_TRAIN_SHA256:
        raise ValueError("training dataset revision; refuse unreviewed comparison")
    if hashlib.sha256(valid_bytes).hexdigest()!=EXPECTED_VALID_SHA256:
        raise ValueError("validation dataset revision; refuse unreviewed comparison")
    if not set(valid_bytes).issubset(set(train_bytes)):
        raise ValueError(f"VALIDATION_OOV={sorted(set(valid_bytes)-set(train_bytes))}; abort not mask")
    if len(train_bytes)<1000 or len(valid_bytes)<1000:
        raise ValueError("insufficient external corpus")
    return train_bytes,valid_bytes,hs

def baseline(train,valid,k,alpha=ALPHA):
    vocab=sorted(set(train)); v=len(vocab)
    histories=collections.defaultdict(collections.Counter)
    totals=collections.Counter()
    for i in range(k,len(train)):
        ctx=train[i-k:i] if k else b""
        histories[ctx][train[i]]+=1
        totals[ctx]+=1
    loss=0.0
    for i in range(CTX,len(valid)):
        ctx=valid[i-k:i] if k else b""
        p=(histories[ctx][valid[i]]+alpha)/(totals[ctx]+alpha*v)
        loss-=math.log(p)
    return loss/(len(valid)-CTX)

def build_and_run(train,valid,epochs,optimized):
    with tempfile.TemporaryDirectory(prefix="niyah-tinystories-") as directory:
        tmp=pathlib.Path(directory)
        (tmp/"src").mkdir();(tmp/"data").mkdir()
        shutil.copy2(ROOT/"src/attention_lm.s",tmp/"src/attention_lm.s")
        (tmp/"data/corpus.txt").write_bytes(train)
        (tmp/"data/heldout.txt").write_bytes(valid)
        binpath=tmp/"attention_lm"
        flags=[f"-Wa,--defsym,EPOCHS={epochs}","-Wa,--defsym,SAMPLE_LEN=0"]
        if optimized:flags.append("-Wa,--defsym,AVX2_AUTO=1")
        subprocess.run(["cc","-nostdlib","-static","-no-pie",*flags,"-o",str(binpath),"src/attention_lm.s"],cwd=tmp,check=True,capture_output=True,timeout=50)
        from time import perf_counter
        start=perf_counter()
        result=subprocess.run([str(binpath)],cwd=tmp,capture_output=True,timeout=120,check=True)
        secs=perf_counter()-start
        if result.stderr:raise RuntimeError(result.stderr[:1000])
        text=result.stdout.decode("ascii")
        def metric(label):
            v=re.search(r"^"+re.escape(label)+r" (\d+)$",text,re.M)
            if not v:raise ValueError(f"missing {label}, output {text[:300]}")
            return int(v.group(1))/1000
        return {"epochs":epochs,"train_nll_nats_byte":metric("final_train_loss_x1000"),"validation_nll_nats_byte":metric("heldout_loss_x1000"),"elapsed_seconds":round(secs,4),"kernel":optimized}

def evaluate(train,valid,story_hashes,epochs):
    values={f"{k}gram_smoothed_nll":baseline(train,valid,k) for k in (0,1,2)}
    no_training=build_and_run(train,valid,0,True)
    trained=build_and_run(train,valid,epochs,True)
    payload={
        "dataset":"roneneldan/TinyStories",
        "dataset_license":"CDLA-Sharing-1.0",
        "source_api":API,
        "train_rows":list(range(TRAIN_N)),
        "validation_rows":list(range(VALID_N)),
        "disjoint_source_splits":True,
        "preprocessing":"concatenate whole stories with two LF bytes; UTF-8 encode then ASCII lowercase",
        "train_bytes":len(train),
        "validation_bytes":len(valid),
        "train_sha256":hashlib.sha256(train).hexdigest(),
        "validation_sha256":hashlib.sha256(valid).hexdigest(),
        "document_source_sha256":story_hashes,
        "ctx_bytes":CTX,
        "model_parameters":9616,
        "notes":["0gram=unigram,1gram=one preceding byte,2gram=two preceding bytes","all smoothed baselines count contexts in the same training bytes only","all validation metrics score indices CTX..N-1 for matched denominators","native model vocabulary is training-byte set, which covers validation bytes","data and bytes NOT committed","lowercase preprocessing is reported; natural language metrics are not instruction following","train+valid concatenated stories mean at most 8 cross-story context boundary tokens per story"],
        "native_untrained":no_training,
        "native_trained":trained,
        "baseline_metrics":values
    }
    payload["native_validation_vs_best_ngram_delta"]=round(trained["validation_nll_nats_byte"]-min(values.values()),6)
    return payload

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=8)
    ap.add_argument("--save",type=pathlib.Path,default=ROOT/"evidence/tinystories-external-local.json")
    args=ap.parse_args()
    if not 1<=args.epochs<=100:raise SystemExit("epochs must be 1..100")
    train,valid,hs=load_documents()
    out=evaluate(train,valid,hs,args.epochs)
    args.save.parent.mkdir(parents=True,exist_ok=True)
    args.save.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()
