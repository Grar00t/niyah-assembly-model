#!/usr/bin/env python3
"""Contract regression for offline local byte files (no network)."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/local_byte_eval.py"
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("local_byte_eval", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def reject(fn, phrase=None):
    try:
        fn()
    except ValueError as e:
        if phrase is not None:
            assert phrase in str(e), (phrase, str(e))
    else:
        raise AssertionError("expected fail-closed ValueError")


def main():
    train = (b"healthy predictable ASCII\n" * 15) + b"back to work\n"
    unseen = (b"known sample with 0123456789\n" * 4)
    assert len(train) > m.CTX and len(unseen) > m.CTX
    assert set(unseen) - set(train)
    result = m.evaluate(train, unseen, epochs=1)
    assert result["scored_positions"] == len(unseen) - m.CTX
    assert result["oov_target_positions"] > 0
    assert result["oov_only_nll_nats_per_byte"]["trigram"] is not None
    assert abs(result["nll_nats_per_byte"]["native_assembly"]
               - result["native_receipt"]["independent_nll"]) < 1e-7
    assert result["train_sha256"] != result["test_sha256"]
    for key in ("train", "test", "data"):
        assert key not in result or not isinstance(result[key], str)
    reject(lambda: m.evaluate(train, train), "identical")
    reject(lambda: m.evaluate(train, unseen, alpha=float("nan")), "alpha")
    reject(lambda: m.evaluate(train, unseen, lam=float("nan")), "lambda")
    reject(lambda: m.evaluate(train, unseen, epochs=101), "epochs")
    assert m.evaluate(train, unseen, epochs=0)["scored_positions"] > 0
    with tempfile.TemporaryDirectory(prefix="niyah-local-contract-") as tmp:
        p = Path(tmp)
        tr, te, output = p/"train.bin", p/"test.bin", p/"report.json"
        tr.write_bytes(train)
        te.write_bytes(unseen)
        ok = subprocess.run([sys.executable, str(SCRIPT),
                             "--train", str(tr), "--test", str(te),
                             "--epochs", "1", "--output", str(output)],
                            capture_output=True, text=True, timeout=120)
        assert ok.returncode == 0, ok.stderr[-700:]
        assert not ok.stdout
        doc = json.loads(output.read_text())
        assert doc["protocol"] == "niyah-local-byte-eval-v1"
        assert len(doc["test_sha256"]) == 64
        assert train.decode() not in output.read_text()
        bad = subprocess.run([sys.executable, str(SCRIPT), "--train", str(tr),
                              "--test", str(te), "--output", str(tr)],
                             capture_output=True, text=True, timeout=10)
        assert bad.returncode != 0
        assert tr.read_bytes() == train
        # No real/user-supplied file is ever an attack target. Both test
        # inputs and the hard links exist only in TemporaryDirectory.
        for label, source in (("train", tr), ("test", te)):
            alias = p / f"{label}-hardlink-report.json"
            before = source.read_bytes()
            os.link(source, alias)
            assert os.path.samefile(alias, source)
            refused = subprocess.run([
                sys.executable, str(SCRIPT),
                "--train", str(tr), "--test", str(te),
                "--epochs", "0", "--output", str(alias),
            ], capture_output=True, text=True, timeout=25)
            assert refused.returncode == 2, refused.stderr[-500:]
            assert "hard link" in refused.stderr, refused.stderr
            assert source.read_bytes() == before, "source file changed"
            assert alias.read_bytes() == before, "hard link inode changed"
            alias.unlink()
        # Normal output overwrites are staged atomically and remain valid.
        output.write_text("stale report", encoding="utf-8")
        again = subprocess.run([
            sys.executable, str(SCRIPT),
            "--train", str(tr), "--test", str(te),
            "--epochs", "0", "--output", str(output),
        ], capture_output=True, text=True, timeout=25)
        assert again.returncode == 0, again.stderr[-500:]
        assert json.loads(output.read_text())["protocol"] == "niyah-local-byte-eval-v1"
        assert tr.read_bytes() == train and te.read_bytes() == unseen
        assert not list(p.glob(".niyah-report-*.tmp"))
        # Reject incompatible byte-passes before any native computation.
        too_large = train * 400
        reject(lambda: m.evaluate(too_large, unseen, epochs=100), "workload")
        large_train = p / "large-train.bin"
        large_train.write_bytes(too_large[:131072])
        budget_report = p / "budget-report.json"
        budget = subprocess.run([
            sys.executable, str(SCRIPT),
            "--train", str(large_train), "--test", str(te),
            "--epochs", "100", "--output", str(budget_report),
        ], capture_output=True, text=True, timeout=10)
        assert budget.returncode == 2
        assert "workload" in budget.stderr, budget.stderr
        assert not budget_report.exists()
        # Enforce a useful fail-closed diagnostic if native is timed out.
        old_native = m.native_true_probs
        def timeout_stub(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="assembly-test", timeout=120)
        try:
            m.native_true_probs = timeout_stub
            try:
                m.evaluate(train, unseen, epochs=1)
            except RuntimeError as exc:
                assert "timed out" in str(exc)
            else:
                raise AssertionError("native timeout escaped cleanly")
        finally:
            m.native_true_probs = old_native
        te.write_bytes(b"")
        empty = subprocess.run([sys.executable, str(SCRIPT), "--train", str(tr),
                                "--test", str(te)],
                               capture_output=True, text=True, timeout=10)
        assert empty.returncode != 0
        assert "evaluation refused" in empty.stderr
        te.write_bytes(unseen)
        link = p/"link.bin"
        link.symlink_to(te)
        reject(lambda: m.file_bytes(link, 1024), "non-symlink")
        reject(lambda: m.file_bytes(te, 16), "exceeds")
    print("PASS: native Assembly full-byte offline file scoring")
    print("PASS: CLI JSON roundtrip and hash-only privacy output")
    print("PASS: train/test self-comparison and invalid arguments fail closed")
    print("PASS: input-overwrite, OOV bytes, symlinks and size limits checked")
    print("PASS: hard-linked output aliases rejected without corrupting disposable inputs")
    print("PASS: atomic unrelated-report replacement and no temporary debris")
    print("PASS: oversized work budget and subprocess timeouts fail closed")


if __name__ == "__main__":
    main()
