#!/usr/bin/env python3
"""Local-only byte-file evaluator; outputs metrics, never raw content."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from hybrid_eval import CTX, blended, native_true_probs, nll, true_probs_ngram


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
    gram = true_probs_ngram(train, test, alpha=alpha, full_byte_vocab=True)
    native, receipt = native_true_probs(train, test, epochs=epochs, full_byte_vocab=True)
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
        train = file_bytes(args.train, args.max_bytes)
        test = file_bytes(args.test, args.max_bytes)
        result = evaluate(train, test, epochs=args.epochs, alpha=args.alpha, lam=args.lambda_weight)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"evaluation refused: {exc}\n")
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        if args.output.is_symlink() or args.output.resolve() in (args.train.resolve(), args.test.resolve()):
            parser.error("output must not overwrite the input files or follow a symlink")
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
