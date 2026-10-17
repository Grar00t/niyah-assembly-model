#!/usr/bin/env python3
"""Local-only byte-file evaluator; outputs metrics, never raw content."""
import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

from hybrid_eval import CTX, blended, native_true_probs, nll, true_probs_ngram


# This is a conservative local-work budget, not a wall-clock SLA. The
# native process also has a fixed timeout; a timeout is reported fail-closed.
MAX_WORK_BYTES = 262144


def reject_output_alias(output, inputs):
    """Detect pathname, symlink and hard-link aliases before any write."""
    if output.is_symlink():
        raise ValueError("output must not be a symlink")
    if output.resolve() in {source.resolve() for source in inputs}:
        raise ValueError("output must not overwrite the input files")
    if output.exists():
        if not output.is_file():
            raise ValueError("output must be a regular file")
        if any(os.path.samefile(output, source) for source in inputs):
            raise ValueError("output must not alias an input hard link")


def write_report_atomic(output, inputs, rendered):
    """Stage on the same filesystem and rename; never truncate shared inodes."""
    reject_output_alias(output, inputs)
    staged = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            dir=output.parent, prefix=".niyah-report-", suffix=".tmp",
            delete=False,
        ) as handle:
            staged = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        # Recheck after preparation; os.replace cannot truncate a hardlink
        # even if an alias is swapped into the path between the two checks.
        reject_output_alias(output, inputs)
        os.replace(staged, output)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def file_bytes(path, max_size):
    item = Path(path)
    if item.is_symlink() or not item.is_file():
        raise ValueError("input must be a regular non-symlink file")
    size = item.stat().st_size
    if size > max_size:
        raise ValueError(f"input exceeds {max_size} bytes")
    with item.open("rb") as f:
        data = f.read(max_size + 1)
    if len(data) > max_size:
        raise ValueError("input grew beyond size limit while reading")
    if len(data) < CTX + 1:
        raise ValueError(f"input must contain at least {CTX + 1} bytes")
    return data


def evaluate(train, test, *, epochs=2, alpha=0.01, lam=0.4):
    if not 0 <= epochs <= 100:
        raise ValueError("epochs must be 0..100")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if not math.isfinite(lam) or not 0 <= lam <= 1:
        raise ValueError("lambda must be finite and within 0..1")
    if train == test:
        raise ValueError("identical train/test bytes; supply an independent test file")
    work_bytes = max(epochs, 1) * len(train) + len(test)
    if work_bytes > MAX_WORK_BYTES:
        raise ValueError(
            f"workload exceeds bounded local budget ({work_bytes} > {MAX_WORK_BYTES} byte-passes); "
            "reduce input sizes or epochs"
        )
    gram = true_probs_ngram(train, test, alpha=alpha, full_byte_vocab=True)
    try:
        native, receipt = native_true_probs(train, test, epochs=epochs, full_byte_vocab=True)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("native execution timed out; reduce input sizes or epochs") from exc
    mix = blended(gram, native, lam)
    known = set(train)
    unseen = [i - CTX for i in range(CTX, len(test)) if test[i] not in known]
    def partial(probs):
        return round(nll([probs[i] for i in unseen]), 9) if unseen else None
    return {
        "protocol": "niyah-local-byte-eval-v1",
        "privacy": "local offline input; only hashes/metrics, no raw bytes",
        "train_sha256": hashlib.sha256(train).hexdigest(),
        "test_sha256": hashlib.sha256(test).hexdigest(),
        "train_bytes": len(train),
        "test_bytes": len(test),
        "scored_positions": len(test) - CTX,
        "oov_target_positions": len(unseen),
        "oov_byte_ids": sorted(set(test[CTX:]) - known),
        "epochs": epochs, "alpha": alpha, "lambda": lam,
        "nll_nats_per_byte": {
            "trigram": round(nll(gram), 9),
            "native_assembly": round(nll(native), 9),
            "hybrid_python": round(nll(mix), 9),
        },
        "oov_only_nll_nats_per_byte": {
            "trigram": partial(gram),
            "native_assembly": partial(native),
            "hybrid_python": partial(mix),
        },
        "native_receipt": receipt,
        "warnings": [
            "Training and scoring are on arbitrary byte files; no meaningful text-generation capability is implied.",
            "Alpha and lambda are fixed arguments, not tuned on the supplied test file.",
            "Raw train/test content is neither uploaded nor included in the JSON.",
            "Use separate, independently collected train and test data; byte inequality does not establish statistical independence.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--lambda-weight", type=float, default=0.4)
    parser.add_argument("--max-bytes", type=int, default=131072)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 9 <= args.max_bytes <= 131072:
        parser.error("--max-bytes must be between 9 and 131072")
    try:
        # Fast refusal before training; check again at the actual write.
        if args.output is not None:
            reject_output_alias(args.output, (args.train, args.test))
        train = file_bytes(args.train, args.max_bytes)
        test = file_bytes(args.test, args.max_bytes)
        result = evaluate(train, test, epochs=args.epochs, alpha=args.alpha, lam=args.lambda_weight)
        rendered = json.dumps(result, indent=2) + "\n"
        if args.output is not None:
            write_report_atomic(args.output, (args.train, args.test), rendered)
        else:
            print(rendered, end="")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"evaluation refused: {exc}\n")


if __name__ == "__main__":
    main()
