#!/usr/bin/env python3
"""Offline differential test for opt-in full-byte Assembly attention evaluation.

Uses an independent count oracle; does not download or train on external data.
"""
import collections
import importlib.util
import math
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("niyah_hybrid", ROOT / "tools/hybrid_eval.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def expect_default_oov_rejection(action):
    try:
        action()
    except ValueError as exc:
        assert "OOV" in str(exc), str(exc)
    else:
        raise AssertionError("default masked model unexpectedly accepted OOV")


def main():
    train = b"the little mouse found cheese. she carried it home.\n" * 8 + b"wanted share friend\n"
    valid = b"the mouse wanted 0123 cheese and fruit.\n" * 3
    assert len(train) > m.CTX and len(valid) > m.CTX
    assert not set(valid).issubset(set(train)), "fixture must genuinely contain OOV bytes"
    expect_default_oov_rejection(lambda: m.true_probs_ngram(train, valid))
    expect_default_oov_rejection(lambda: m.native_true_probs(train, valid, epochs=2))

    for bad_alpha in (0.0, -0.1, float("nan"), float("inf")):
        try:
            m.true_probs_ngram(train, valid, alpha=bad_alpha, full_byte_vocab=True)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid alpha must reject")
    full_gram = m.true_probs_ngram(train, valid, full_byte_vocab=True)
    triples = collections.Counter(zip(train[:-2], train[1:-1], train[2:]))
    contexts = collections.Counter(zip(train[:-2], train[1:-1]))
    reference = [
        (triples[(valid[i - 2], valid[i - 1], valid[i])] + 0.5) /
        (contexts[(valid[i - 2], valid[i - 1])] + 128.0)
        for i in range(m.CTX, len(valid))
    ]
    assert len(reference) == len(full_gram)
    assert max(abs(a - b) for a, b in zip(reference, full_gram)) < 1e-14

    native, receipt = m.native_true_probs(train, valid, epochs=2, full_byte_vocab=True)
    assert len(native) == len(valid) - m.CTX
    assert all(math.isfinite(v) and 0 < v <= 1 for v in native)
    assert abs(m.nll(native) - receipt["native_reported_nll"]) <= 0.001
    assert m.blended(full_gram, native, 0) == full_gram
    assert m.blended(full_gram, native, 1) == native
    assert all(0 < q <= 1 for q in m.blended(full_gram, native, 0.4))

    with tempfile.TemporaryDirectory(prefix="niyah-fullbyte-guard-") as folder:
        root = Path(folder)
        (root / "src").mkdir()
        (root / "data").mkdir()
        (root / "models").mkdir()
        (root / "src/attention_lm.s").write_bytes((ROOT / "src/attention_lm.s").read_bytes())
        (root / "data/corpus.txt").write_bytes(train)
        (root / "data/heldout.txt").write_bytes(valid)
        # Fail closed: v2 checkpoint packaging does NOT encode a vocabulary policy.
        for extra in ["INFERENCE=1", "EXPORT_WEIGHTS=1"]:
            cmd = ["cc", "-nostdlib", "-static", "-no-pie",
                   "-Wa,--defsym,FULL_BYTE_VOCAB=1", f"-Wa,--defsym,{extra}",
                   "-o", str(root / "invalid"), "src/attention_lm.s"]
            proc = subprocess.run(cmd, cwd=root, capture_output=True, timeout=45)
            assert proc.returncode != 0, f"{extra} bypassed checkpoint policy guard"
            assert b"FULL_BYTE_VOCAB" in proc.stderr, proc.stderr[:1000]

    print("PASS: masked legacy path rejects unseen bytes")
    print("PASS: 256-byte additive-smoothing n-gram matches independent count oracle")
    print("PASS: native GNU x86-64 Assembly exports valid OOV true-byte probabilities")
    print("PASS: full-byte stream recreates native NLL, fixed mixture boundaries")
    print("PASS: full-byte mode cannot silently share v2 checkpoint/export semantics")


if __name__ == "__main__":
    main()
