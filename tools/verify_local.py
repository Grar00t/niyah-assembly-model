from pathlib import Path
import array
import hashlib
import json
import math
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

root = Path(__file__).resolve().parents[1]
out = Path(tempfile.mkdtemp(prefix="niyah-proof-", dir=root.parent))
report = {
    "status": "FAIL",
    "project": str(root),
    "machine": platform.machine(),
    "operations": [],
    "memory_safety_proven": False,
}

def require(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path

def sha(path):
    h = hashlib.sha256()
    with require(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def run(name, command, timeout=300):
    command = list(map(str, command))
    print(f"\n=== {name} ===", flush=True)
    print(shlex.join(command), flush=True)
    item = {"name": name, "command": command}
    report["operations"].append(item)
    started = time.monotonic()
    try:
        process = subprocess.run(
            command, cwd=root, capture_output=True, timeout=timeout
        )
        stdout, stderr = process.stdout, process.stderr
        item["exit_code"] = process.returncode
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or b"", error.stderr or b""
        item["status"] = "TIMEOUT"
        raise
    except OSError as error:
        stdout, stderr = b"", str(error).encode()
        item["status"] = "START_FAILED"
        raise
    finally:
        item["seconds"] = round(time.monotonic() - started, 6)
        (out / f"{name}.stdout").write_bytes(stdout)
        (out / f"{name}.stderr").write_bytes(stderr)
        for data in (stdout, stderr):
            if data:
                print(data.decode("utf-8", errors="backslashreplace"))
    if process.returncode:
        item["status"] = "FAIL"
        raise RuntimeError(f"{name}: exit {process.returncode}")
    item["status"] = "PASS"
    return stdout

def test_report():
    data = json.loads(require(root / "verification_results.json").read_text())
    tests = data.get("tests")
    if not isinstance(tests, list) or not tests:
        raise ValueError("Missing test results")
    if not all(test.get("pass") is True for test in tests):
        raise ValueError("A test failed")
    if data.get("passed") != len(tests) or data.get("total") != len(tests):
        raise ValueError("Inconsistent test counts")
    return data

try:
    required = [
        "src/trigram.s",
        "data/corpus.txt",
        "Makefile",
        "tests/test_generation.py",
        "models/weights.f32",
        "models/checkpoint.json",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing files: " + ", ".join(missing))

    for tool in ("cc", "make"):
        if shutil.which(tool) is None:
            raise FileNotFoundError(f"Required executable missing: {tool}")

    source = root / "src/trigram.s"
    corpus = root / "data/corpus.txt"
    weights = root / "models/weights.f32"
    inputs = {
        name: sha(root / name)
        for name in ("src/trigram.s", "data/corpus.txt", "Makefile")
    }
    report["inputs"] = inputs
    corpus.read_bytes().decode("utf-8", errors="strict")

    print("PROJECT:", root)
    print("REPORTS:", out)
    print("\n=== MAKEFILE ===\n" + (root / "Makefile").read_text())

    run("make-plan", ["make", "-n", "-B"])

    executable = out / "trigram"
    run("compile", [
        "cc", "-nostdlib", "-static", "-no-pie",
        "-o", executable, source,
    ])

    with require(executable).open("rb") as stream:
        header = stream.read(20)
    if (
        len(header) != 20
        or header[:6] != b"\x7fELF\x02\x01"
        or int.from_bytes(header[18:20], "little") != 62
    ):
        raise ValueError("Expected little-endian x86-64 ELF64")
    report["binary_sha256"] = sha(executable)

    trained = run("training", [executable])

    previous = root / "verification_results.json"
    if previous.is_file():
        shutil.copyfile(previous, out / "previous-test-results.json")

    run("package-tests", [sys.executable, root / "tests/test_generation.py"])
    report["package_tests"] = test_report()

    runner = Path("/home/a/trigram-py67LL90/verify_model.py")
    lexer = Path("/home/a/niyah-input-fix-xbgJg8lR/niyahlex")
    require(runner)
    require(lexer)

    run("extended-tests", [
        sys.executable, runner, source, "--lexer", lexer,
    ])
    report["extended_tests"] = test_report()

    manifest = json.loads((root / "models/checkpoint.json").read_text())
    if weights.stat().st_size != 67108864:
        raise ValueError("Checkpoint size mismatch")
    if sha(weights) != manifest["weights_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")

    expected = manifest.get("files")
    if expected is None:
        expected = {
            "src/trigram.s": manifest["source_sha256"],
            "data/corpus.txt": manifest["corpus_sha256"],
        }
    if expected != {
        name: inputs[name] for name in ("src/trigram.s", "data/corpus.txt")
    }:
        raise ValueError("Checkpoint source/corpus identity mismatch")

    with weights.open("rb") as stream:
        while chunk := stream.read(1048576):
            values = array.array("f")
            if values.itemsize != 4:
                raise ValueError("Unsupported host float size")
            values.frombytes(chunk)
            if sys.byteorder != "little":
                values.byteswap()
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Checkpoint contains NaN or Inf")

    report["weights_sha256"] = sha(weights)

    inference = out / "generate"
    run("compile-inference", [
        "cc", "-nostdlib", "-static", "-no-pie",
        "-Wa,--defsym,INFERENCE=1", "-o", inference, source,
    ])
    restored = run("inference", [inference], timeout=60)

    marker = b"--- sample ---\n"
    if trained.count(marker) != 1 or restored.count(marker) != 1:
        raise ValueError("Invalid sample markers")
    sample = restored.split(marker, 1)[1]
    if trained.split(marker, 1)[1] != sample:
        raise ValueError("Checkpoint replay mismatch")
    report["checkpoint_replay"] = "PASS"

    try:
        sample.decode("utf-8", errors="strict")
        report["sample_utf8"] = "valid"
    except UnicodeDecodeError:
        report["sample_utf8"] = "invalid"

    for name, expected_hash in inputs.items():
        if sha(root / name) != expected_hash:
            raise ValueError(f"File changed during verification: {name}")
    if sha(weights) != report["weights_sha256"]:
        raise ValueError("Weights changed during verification")

    report["status"] = "PASS"

except Exception as error:
    report["error"] = {
        "type": type(error).__name__,
        "message": str(error),
    }
    print(f"\nFAIL: {type(error).__name__}: {error}", file=sys.stderr)

finally:
    receipt = out / "proof.json"
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    print("\n=== FINAL ===")
    print("STATUS:", report["status"])
    print("RECEIPT:", receipt)
    print("LOGS:", out)

raise SystemExit(0 if report["status"] == "PASS" else 1)
