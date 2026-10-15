#!/usr/bin/env python3
"""Checkpoint packaging/verification for the assembly attention model.
No Python is loaded by the model at training or inference time.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/"src"/"attention_lm.s"
TRAIN = ROOT/"data"/"corpus.txt"
VALID = ROOT/"data"/"heldout.txt"
MODELS = ROOT/"models"
WEIGHTS = MODELS/"attention_weights.f32"
MANIFEST = MODELS/"attention_manifest.json"
EPOCHS = 240
WEIGHT_BYTES = 38464

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build(exe, **opts):
    flags = [f"-Wa,--defsym,{key}={val}" for key,val in opts.items()]
    subprocess.run(["cc","-nostdlib","-static","-no-pie",*flags,
                    "-o",str(exe),str(MODEL)],cwd=ROOT,
                    check=True,capture_output=True,timeout=60)

def run(exe):
    proc = subprocess.run([str(exe)],check=True,capture_output=True,timeout=180)
    if proc.stderr:
        raise RuntimeError(f"unexpected stderr: {proc.stderr[:250]!r}")
    return proc.stdout

def validate_weight_bytes(data):
    if len(data)!=WEIGHT_BYTES:
        raise ValueError(f"invalid weight count: {len(data)}")
    for (weight,) in struct.iter_unpack("<f",data):
        if not math.isfinite(weight):
            raise ValueError("non-finite checkpoint weight")
    return hashlib.sha256(data).hexdigest()

def train():
    MODELS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        exe=Path(td)/"train"
        build(exe,EPOCHS=EPOCHS,SAMPLE_LEN=0,EXPORT_WEIGHTS=1)
        output=run(exe)
        if b"heldout_loss_x1000 " not in output[:-WEIGHT_BYTES]:
            raise ValueError("training evidence header absent")
        raw=output[-WEIGHT_BYTES:]
        sha=validate_weight_bytes(raw)
    evidence=output[:-WEIGHT_BYTES:].decode("utf-8",errors="strict")
    summary={
        "format":"niyah-attention-f32-v2-learned-qkv",
         "architecture":"D=16, CTX=8, trainable WQ/WK/WV, one-head attention, residual ReLU FFN",
        "epoch_count":EPOCHS,
        "weights_len":WEIGHT_BYTES,
        "sha256_weights":sha,
        "sha256_assembly":digest(MODEL),
        "sha256_train_corpus":digest(TRAIN),
        "sha256_heldout_corpus":digest(VALID),
        "stdout_metrics":evidence.strip().splitlines(),
        "warnings":["SHA-256 is integrity, not authenticity",
                    "no optimizer state; individual file replacements are not a transaction",
                    "training corpus is too small to establish broad generalization"]
    }
    fd,temp=tempfile.mkstemp(dir=MODELS,prefix=".attention-",suffix=".tmp")
    try:
        with os.fdopen(fd,"wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp,WEIGHTS)
    finally:
        if os.path.exists(temp): os.unlink(temp)
    fd,temp=tempfile.mkstemp(dir=MODELS,prefix=".manifest-",suffix=".tmp")
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            json.dump(summary,f,indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp,MANIFEST)
    finally:
        if os.path.exists(temp): os.unlink(temp)
    print(f"PASS checkpoint train sha256={sha} bytes={len(raw)}")

def verify():
    if not WEIGHTS.exists() or not MANIFEST.exists():
        raise FileNotFoundError("no saved attention checkpoint: run make attention-checkpoint")
    info=json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual=validate_weight_bytes(WEIGHTS.read_bytes())
    checks={
      "sha256_weights":actual,
      "sha256_assembly":digest(MODEL),
      "sha256_train_corpus":digest(TRAIN),
      "sha256_heldout_corpus":digest(VALID)
    }
    if info.get("format")!="niyah-attention-f32-v2-learned-qkv" or info.get("weights_len")!=WEIGHT_BYTES:
        raise ValueError("unrecognized checkpoint format")
    for name,value in checks.items():
        if info.get(name)!=value:
            raise ValueError(f"checkpoint mismatch for {name}")
    print(f"PASS attention checkpoint verified sha256={actual}")

def generate(seed=42,sample=64,auto=False):
    # Check manifest/source/weights BEFORE compiling an .incbin inference binary.
    verify()
    with tempfile.TemporaryDirectory() as td:
        exe=Path(td)/"infer"
        opts={"INFERENCE":1,"SEED":seed,"SAMPLE_LEN":sample}
        if auto:
            opts["AVX2_AUTO"]=1
        build(exe,**opts)
        output=run(exe)
        if auto:
            prefix=b"kernel_avx2 "
            if len(output)<len(prefix)+2 or not output.startswith(prefix) or output[len(prefix)] not in b"01":
                raise ValueError("invalid runtime kernel dispatch receipt")
            if output[len(prefix)+1:len(prefix)+2]!=b"\n":
                raise ValueError("malformed CPU dispatch line")
            output=output[len(prefix)+2:]
        marker=b"--- attention sample ---\n"
        if not output.startswith(marker) or not output.endswith(b"\n"):
            raise ValueError("missing inference marker/newline")
        if len(output[len(marker):-1])!=8+sample:
            raise ValueError("incorrect inference sample length")
        return output

def replay():
    generated=generate()
    with tempfile.TemporaryDirectory() as td:
        exe=Path(td)/"train"
        build(exe,EPOCHS=EPOCHS,SEED=42,SAMPLE_LEN=64)
        trained=run(exe)
    train_sample=trained.split(b"--- attention sample ---\n",1)[1]
    infer_sample=generated.split(b"--- attention sample ---\n",1)[1]
    if train_sample!=infer_sample:
        raise ValueError("inference did not replay the trained weights exactly")
    print("PASS exact train -> saved weights -> inference replay")

if __name__=="__main__":
    if len(sys.argv)!=2 or sys.argv[1] not in (
            "train","verify","generate","generate-auto","test"):
        raise SystemExit("usage: checkpoint_attention.py train|verify|generate|generate-auto|test")
    try:
        if sys.argv[1]=="train":
            train()
        elif sys.argv[1]=="verify":
            verify()
        elif sys.argv[1]=="generate":
            sys.stdout.buffer.write(generate())
        elif sys.argv[1]=="generate-auto":
            sys.stdout.buffer.write(generate(auto=True))
        else:
            replay()
    except (ValueError,FileNotFoundError,RuntimeError,
            subprocess.CalledProcessError,subprocess.TimeoutExpired) as e:
        print(f"FAIL: {e}",file=sys.stderr)
        raise SystemExit(1)
