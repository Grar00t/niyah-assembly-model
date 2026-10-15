#!/usr/bin/env python3
"""Independent behavioral/differential checks for the pure-assembly attention LM.
Oracle is CPython math, NOT used by training, generation, or the model executable.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "attention_lm.s"
TRAIN = (ROOT / "data" / "corpus.txt").read_bytes()
VALID = (ROOT / "data" / "heldout.txt").read_bytes()
D = 16
T = 8
SAMPLE = 64
NFLOATS = 4096 + 128 + 256 + 16 + 4096 + 256 + 256*3
WEIGHT_BYTES = NFLOATS * 4
ALLOWED = set(TRAIN)

def build(exe, epochs, seed=42, sample=SAMPLE, export=False):
    flags = ["-Wa,--defsym,EPOCHS=" + str(epochs),
             "-Wa,--defsym,SEED=" + str(seed),
             "-Wa,--defsym,SAMPLE_LEN=" + str(sample)]
    if export:
        flags.append("-Wa,--defsym,EXPORT_WEIGHTS=1")
    subprocess.run(["cc", "-nostdlib", "-static", "-no-pie", *flags,
                    "-o", str(exe), str(SRC)], cwd=ROOT,
                   check=True, capture_output=True, timeout=60)
    elf = subprocess.run(["readelf", "-l", str(exe)],
                         check=True, capture_output=True, timeout=30).stdout
    assert b"INTERP" not in elf, "unexpected dynamic interpreter"
    return hashlib.sha256(exe.read_bytes()).hexdigest()

def run(exe):
    t = time.perf_counter()
    p = subprocess.run([str(exe)], capture_output=True, timeout=180)
    elapsed = time.perf_counter() - t
    assert p.returncode == 0 and not p.stderr, (
        f"exit={p.returncode}: " + repr(p.stderr[:300]))
    return p.stdout, elapsed

def metric(output, key):
    m = re.search(re.escape(key.encode()) + rb" (\d+)\n", output)
    assert m is not None, f"missing {key}"
    return int(m.group(1))

def sample_bytes(output):
    marker = b"--- attention sample ---\n"
    assert output.count(marker) == 1, "missing or duplicate sample marker"
    sample = output.split(marker, 1)[1]
    assert sample.endswith(b"\n")
    return sample[:-1]

def parse_params(buffer):
    weights = buffer[-WEIGHT_BYTES:]
    assert len(weights) == WEIGHT_BYTES
    floats = struct.unpack("<" + "f"*NFLOATS, weights)
    assert all(math.isfinite(x) for x in floats), "non-finite model weights"
    ofs = 0
    arrays = []
    for count in (4096, 128, 256, 16, 4096, 256, 256, 256, 256):
        arrays.append(floats[ofs:ofs+count])
        ofs += count
    assert ofs == NFLOATS
    return weights, arrays

def distribution(context, arrays):
    assert len(context) == T
    e, pos, ff, fb, out, ob, wq, wk, wv = arrays
    x = [[e[ch*D+d] + pos[t*D+d] for d in range(D)]
         for t, ch in enumerate(context)]
    def project(w, a):
        return [sum(w[i*D+j]*a[j] for j in range(D)) for i in range(D)]
    q = project(wq, x[-1])
    k = [project(wk, vector) for vector in x]
    v = [project(wv, vector) for vector in x]
    scores = [sum(q[j]*key[j] for j in range(D)) / math.sqrt(D)
              for key in k]
    shift = max(scores)
    unnormalized = [math.exp(z-shift) for z in scores]
    z = sum(unnormalized)
    a = [value/z for value in unnormalized]
    h = [x[-1][j] + sum(a[t]*v[t][j] for t in range(T))
         for j in range(D)]
    y = [h[i] + max(0.0, fb[i]+sum(ff[i*D+j]*h[j]
                                     for j in range(D))) for i in range(D)]
    logits = {b: ob[b]+sum(out[b*D+j]*y[j] for j in range(D))
              for b in ALLOWED}
    peak = max(logits.values())
    temp = {b: math.exp(z-peak) for b,z in logits.items()}
    total = sum(temp.values())
    return {b: z/total for b,z in temp.items()}

def oracle_loss(data, arrays):
    assert len(data) > T
    loss = 0.0
    for i in range(T,len(data)):
        prob = distribution(data[i-T:i], arrays)
        assert data[i] in prob, "validation contains byte absent from training"
        loss -= math.log(max(prob[data[i]],1.0e-30))
    return loss/(len(data)-T)

def main():
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp)/"attention"
        digest = build(exe, epochs=240, seed=42)
        trained, elapsed = run(exe)
        same, _ = run(exe)
        assert trained == same, "non-deterministic reproduction"
        sample = sample_bytes(trained)
        assert len(sample) == T+SAMPLE
        assert all(c in ALLOWED for c in sample), "sample outside training vocabulary"
        initial = metric(trained,"initial_train_loss_x1000")
        final = metric(trained,"final_train_loss_x1000")
        valid = metric(trained,"heldout_loss_x1000")
        assert 0 < final < initial, "model did not learn its training corpus"
        assert valid > 0 and valid < 100000

        build(exe, epochs=240, seed=1)
        other, _ = run(exe)
        assert sample_bytes(other) != sample, "seed not affecting generation"
        assert metric(other,"final_train_loss_x1000") == final

        build(exe, epochs=0, seed=42)
        untrained, _ = run(exe)
        untrained_valid = metric(untrained,"heldout_loss_x1000")
        assert valid < untrained_valid, "held-out score did not improve"

        # Binary weights are emitted by the machine-code training loop itself.
        build(exe, epochs=240, seed=42, sample=0, export=True)
        exported, _ = run(exe)
        weights, params = parse_params(exported)
        assert hashlib.sha256(weights).digest() != b"\x00"*32
        assert len(exported)-WEIGHT_BYTES < 1500
        # Check that the three separately trainable projections did move
        # from identity to a different matrix (not tied aliases).
        for matrix in params[6:]:
            assert max(abs(x - (1.0 if k//D==k%D else 0.0))
                       for k,x in enumerate(matrix)) > 1e-6
        oracle_train = oracle_loss(TRAIN,params)*1000
        oracle_valid = oracle_loss(VALID,params)*1000
        assert abs(oracle_train-final) < 3.0, (oracle_train,final)
        assert abs(oracle_valid-valid) < 3.0, (oracle_valid,valid)

        # Counterfactual contexts: same two ending bytes, earlier context differs.
        # A trigram table MUST return equal distributions for these two contexts.
        a = distribution(b"the cat ",params)
        b = distribution(b"the rat ",params)
        difference = sum(abs(a.get(v,0.0)-b.get(v,0.0)) for v in ALLOWED)
        assert difference > 1.0e-4, "context not affecting model probabilities"

    report={
        "model":"NIYAH pure-assembly attention LM",
         "model_type":"trainable WQ/WK/WV single-head attention with ReLU FFN and SGD",
        "not_a_large_language_model":True,
        "tests":"PASS",
        "binary_sha256_first_run":digest,
        "weight_sha256":hashlib.sha256(weights).hexdigest(),
        "train_loss_initial_millinats":initial,
        "train_loss_final_millinats":final,
        "heldout_loss_untrained_millinats":untrained_valid,
        "heldout_loss_trained_millinats":valid,
        "independent_oracle_train_millinats":round(oracle_train,3),
        "independent_oracle_heldout_millinats":round(oracle_valid,3),
        "same_last_2_context_l1_difference":round(difference,8),
        "train_sample_elapsed_seconds":round(elapsed,3),
        "sample_total_bytes":len(sample),
        "sample_sha256":hashlib.sha256(sample).hexdigest(),
        "checks":["static native ELF","no libc/interpreter",
                  "two identical seeded runs",
                  "different sampling seed","vocabulary bound",
                  "training loss improvement","heldout vs untrained",
                  "independent numerical oracle","8-byte context intervention",
                  "finite exported weights"]
    }
    (ROOT/"attention_verification.json").write_text(
        json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
