#!/usr/bin/env python3
"""OOV diagnostic, NOT blind heldout: 256 shared byte IDs and calibrated baselines.

The validation 200..207 OOV slice was inspected on 2026-10-16 before this
support was added. Choose count-model smoothing and interpolation solely
from the older validation 0..5 calibration corpus.
No dataset bytes, native weights, or secrets are committed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

import hybrid_eval as h

ROOT = Path(__file__).resolve().parents[1]
OFFSET = 200
COUNT = 8
EXPECTED_SHA256 = "57c7ceb60e460aaf9d69f3ec58f163f81055195f07268fefe9566ad7f8b0975a"
EXPECTED_LENGTH = 12381
ALPHA_GRID = (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)


def load_locked_oov():
    query = urllib.parse.urlencode({
        "dataset": h.source.DATASET, "config": "default",
        "split": "validation", "offset": OFFSET, "length": COUNT})
    req = urllib.request.Request(h.source.API + "?" + query, headers={
        "User-Agent": "niyah-assembly-fullbyte-oov-revisit/1.0"})
    with urllib.request.urlopen(req, timeout=45) as response:
        rows = json.load(response)["rows"]
    if len(rows) != COUNT or [r["row_idx"] for r in rows] != list(range(OFFSET, OFFSET + COUNT)):
        raise ValueError("locked OOV slice: wrong row IDs")
    stories = [r["row"]["text"] for r in rows]
    corpus = ("\n\n".join(stories) + "\n").encode().lower()
    digest = hashlib.sha256(corpus).hexdigest()
    if len(corpus) != EXPECTED_LENGTH or digest != EXPECTED_SHA256:
        raise ValueError("locked OOV slice: bytes/hash changed; refuse to score")
    return corpus, [hashlib.sha256(s.encode()).hexdigest() for s in stories]


def subset_nll(probs, indices):
    return h.nll([probs[i] for i in indices])


def experiment(epochs):
    if not 0 <= epochs <= 100:
        raise ValueError("epochs must be 0..100")
    train, calibration, train_dev_hashes = h.source.load_documents()
    test, test_hashes = load_locked_oov()
    if set(train_dev_hashes) & set(test_hashes):
        raise ValueError("duplicate exact source stories")
    train_set = set(train)
    oov_ids = sorted(set(test) - train_set)
    if not oov_ids:
        raise ValueError("locked test no longer contains OOV bytes")

    # Freeze ALL hyperparameter decisions before scoring OOV test rows.
    cal_neural, cal_receipt = h.native_true_probs(
        train, calibration, epochs, full_byte_vocab=True)
    alpha_calibration = {
        str(alpha): round(h.nll(h.true_probs_ngram(
            train, calibration, alpha=alpha, full_byte_vocab=True)), 9)
        for alpha in ALPHA_GRID
    }
    selected_alpha = min(ALPHA_GRID, key=lambda a: (alpha_calibration[str(a)], a))
    cal_gram = h.true_probs_ngram(
        train, calibration, alpha=selected_alpha, full_byte_vocab=True)
    lambda_calibration = {
        str(weight): round(h.nll(h.blended(cal_gram, cal_neural, weight)), 9)
        for weight in h.LAMBDA_GRID
    }
    selected_lambda = min(
        h.LAMBDA_GRID, key=lambda w: (lambda_calibration[str(w)], w))

    test_neural, test_receipt = h.native_true_probs(
        train, test, epochs, full_byte_vocab=True)
    test_gram = h.true_probs_ngram(
        train, test, alpha=selected_alpha, full_byte_vocab=True)
    test_mix = h.blended(test_gram, test_neural, selected_lambda)
    naive_gram = h.true_probs_ngram(
        train, test, alpha=0.5, full_byte_vocab=True)
    # Native probabilities are ordered by true target position, starting at CTX.
    oov_positions = [i - h.CTX for i in range(h.CTX, len(test)) if test[i] not in train_set]
    assert oov_positions, "no scored OOV positions"

    return {
        "protocol": "2026-10-16-revisit-observed-failing-OOV-slice",
        "not_a_blind_heldout_result": True,
        "dataset": "roneneldan/TinyStories",
        "license": "CDLA-Sharing-1.0",
        "source_api": h.source.API,
        "train_rows": "train 0..11",
        "calibration_rows": "validation 0..5; previously measured",
        "oov_test_rows": "validation 200..207; previously examined OOV slice",
        "neural_runtime": "x86-64 GNU Assembly, native learned Q/K/V, full-byte mask",
        "ngram_and_blend_runtime": "Python external evaluator, not Assembly generation",
        "all_byte_support": 256,
        "legacy_v2_checkpoint_compatible": False,
        "epochs": epochs,
        "alpha_grid": list(ALPHA_GRID),
        "alpha_calibration_nll_grid": alpha_calibration,
        "alpha_selected_from_calibration_only": selected_alpha,
        "lambda_grid": list(h.LAMBDA_GRID),
        "lambda_calibration_nll_grid": lambda_calibration,
        "lambda_selected_from_calibration_only": selected_lambda,
        "train_sha256": hashlib.sha256(train).hexdigest(),
        "calibration_sha256": hashlib.sha256(calibration).hexdigest(),
        "oov_test_sha256": hashlib.sha256(test).hexdigest(),
        "oov_test_source_doc_sha256": test_hashes,
        "train_bytes": len(train),
        "calibration_bytes": len(calibration),
        "oov_test_bytes": len(test),
        "oov_byte_ids": oov_ids,
        "scored_oov_positions": len(oov_positions),
        "scored_total_positions": len(test) - h.CTX,
        "calibration_native_receipt": cal_receipt,
        "test_native_receipt": test_receipt,
        "naive_alpha_0_5_trigram_nll": round(h.nll(naive_gram), 9),
        "calibrated_trigram_nll": round(h.nll(test_gram), 9),
        "fullbyte_native_nll": round(h.nll(test_neural), 9),
        "calibrated_hybrid_nll": round(h.nll(test_mix), 9),
        "calibrated_hybrid_minus_calibrated_trigram": round(
            h.nll(test_mix) - h.nll(test_gram), 9),
        "calibrated_hybrid_beats_calibrated_trigram": h.nll(test_mix) < h.nll(test_gram),
        "oov_only_trigram_nll": round(subset_nll(test_gram, oov_positions), 9),
        "oov_only_neural_nll": round(subset_nll(test_neural, oov_positions), 9),
        "oov_only_hybrid_nll": round(subset_nll(test_mix, oov_positions), 9),
        "metric": "mean true-next-byte -log(p), nats/byte; valid indices CTX..N-1",
        "limitations": [
            "Not blind. The OOV slice was previously inspected, not hyperparameter selected.",
            "Only 8 OOV target positions; full-set aggregate is dominated by seen bytes.",
            "Alpha and lambda picked on previously measured 6-document calibration set.",
            "One seed and 12 training stories; near-duplicate contents possible.",
            "Full-byte trained checkpoints require a new manifest protocol.",
            "No end-to-end pure-Assembly count-hybrid inference, semantic competence or user benefit established."
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--save", type=Path, default=ROOT / "evidence/fullbyte-oov-revisit-local.json")
    args = p.parse_args()
    result = experiment(args.epochs)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k != "oov_test_source_doc_sha256"}, indent=2))


if __name__ == "__main__":
    main()
