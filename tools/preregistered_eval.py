#!/usr/bin/env python3
"""Pre-registered, story-boundary-aware OOV scoring for a native Assembly LM.

This source fixes the test row IDs, hyperparameter grids, runtime, scoring
rules, and bootstrap before fetching the test split. Never tune after looking
at the test results. No third-party story text is written to the repository.
"""
import argparse
import hashlib
import json
import math
import pathlib
import random
import urllib.parse
import urllib.request

import hybrid_eval as h

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTOCOL = "NIYAH-2026-10-17-preregistered-story-OOV-v1"
DATASET = "roneneldan/TinyStories"
TEST_SPLIT = "validation"
TEST_OFFSET = 1200
TEST_COUNT = 24
EPOCHS = 32
CONTEXT_BYTES = h.CTX
NGRAM_ORDERS = (0, 1, 2)
ALPHAS = (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
LAMBDAS = h.LAMBDA_GRID
BOOTSTRAP_SEED = 20261017
BOOTSTRAP_REPEATS = 2000
MAX_TOTAL_BYTES = 300000
EXPECTED_TRAIN_SHA256 = h.source.EXPECTED_TRAIN_SHA256
EXPECTED_CAL_SHA256 = h.source.EXPECTED_VALID_SHA256


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def normalize_and_span(stories):
    """Returns normalized bytes and [start, end) spans excluding separators.

    Only score target positions after CONTEXT_BYTES within the SAME story,
    so delimiter and cross-story context leakage do not affect comparison.
    """
    if not stories or any(not isinstance(t, str) for t in stories):
        raise ValueError("expected non-empty list of textual stories")
    result = bytearray()
    spans = []
    for i, story in enumerate(stories):
        blob = story.encode("utf-8").lower()
        if len(blob) <= CONTEXT_BYTES:
            raise ValueError("story too short for within-story contexts")
        start = len(result)
        result.extend(blob)
        spans.append((start, len(result)))
        result.extend(b"\n\n" if i + 1 < len(stories) else b"\n")
        if len(result) > MAX_TOTAL_BYTES:
            raise ValueError("scoring corpus exceeds fixed byte budget")
    return bytes(result), spans


def validated_rows(payload):
    """Parse official HF row API response with exact predeclared row IDs."""
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError("HF rows response malformed")
    rows = payload["rows"]
    ids = [r.get("row_idx") if isinstance(r, dict) else None for r in rows]
    if ids != list(range(TEST_OFFSET, TEST_OFFSET + TEST_COUNT)):
        raise ValueError("unexpected test rows; refuse substitution")
    stories = []
    for row in rows:
        value = row.get("row")
        if not isinstance(value, dict) or not isinstance(value.get("text"), str):
            raise ValueError("missing text in locked test row")
        stories.append(value["text"])
    if len({sha256(s.encode("utf-8")) for s in stories}) != TEST_COUNT:
        raise ValueError("duplicate exact test documents")
    return stories


def fetch_test():
    params = urllib.parse.urlencode({
        "dataset": DATASET, "config": "default", "split": TEST_SPLIT,
        "offset": TEST_OFFSET, "length": TEST_COUNT,
    })
    req = urllib.request.Request(
        h.source.API + "?" + params,
        headers={"User-Agent": "NIYAH-preregistered-eval/1.0"})
    with urllib.request.urlopen(req, timeout=45) as response:
        payload = json.load(response)
    return validated_rows(payload)


def calibrate(train, calibration, native_predictor=None):
    """All model selection uses previously published calibration rows 0..5."""
    native_predictor = native_predictor or h.native_true_probs
    if sha256(train) != EXPECTED_TRAIN_SHA256 or sha256(calibration) != EXPECTED_CAL_SHA256:
        raise ValueError("train/calibration input changed")
    neural, receipt = native_predictor(
        train, calibration, epochs=EPOCHS, full_byte_vocab=True)
    grid = {}
    candidates = []
    for order in NGRAM_ORDERS:
        for alpha in ALPHAS:
            probs = h.true_probs_ngram(
                train, calibration, k=order, alpha=alpha, full_byte_vocab=True)
            loss = h.nll(probs)
            grid[f"{order}:{alpha}"] = round(loss, 9)
            candidates.append((loss, order, alpha))
    _, best_k, best_alpha = min(candidates)
    gram = h.true_probs_ngram(
        train, calibration, k=best_k, alpha=best_alpha, full_byte_vocab=True)
    mixture_grid = {
        str(w): round(h.nll(h.blended(gram, neural, w)), 9)
        for w in LAMBDAS
    }
    best_lambda = min(LAMBDAS, key=lambda w: (mixture_grid[str(w)], w))
    return {
        "n_gram_order": best_k,
        "alpha": best_alpha,
        "lambda": best_lambda,
        "count_calibration_grid": grid,
        "mixture_calibration_grid": mixture_grid,
        "calibration_native_receipt": receipt,
    }


def story_indices(spans):
    return [list(range(start + CONTEXT_BYTES, end)) for start, end in spans]


def summarize_scored(train, test, spans, native_probs, gram_probs, weight):
    """Evaluate aligned true-byte probabilities, excluding all story boundaries."""
    expected = len(test) - CONTEXT_BYTES
    if len(native_probs) != expected or len(gram_probs) != expected:
        raise ValueError("native/count probability streams have wrong length")
    mix = h.blended(gram_probs, native_probs, weight)
    seen = set(train)
    records = []
    component = ("count", "native", "hybrid")
    aggregate = {name: {"n": 0, "loss_sum": 0.0} for name in component}
    by_oov = {
        cat: {name: {"n": 0, "loss_sum": 0.0} for name in component}
        for cat in ("oov", "seen")
    }
    for indices in story_indices(spans):
        rec = {"scored_positions": len(indices), "oov_targets": 0,
               "component_loss_sums": {name: 0.0 for name in component}}
        for pos in indices:
            j = pos - CONTEXT_BYTES
            cat = "seen" if test[pos] in seen else "oov"
            if cat == "oov":
                rec["oov_targets"] += 1
            for name, probs in (("count", gram_probs), ("native", native_probs),
                                ("hybrid", mix)):
                p = probs[j]
                if not math.isfinite(p) or p <= 0 or p > 1:
                    raise ValueError("invalid probability in scored stream")
                loss = -math.log(p)
                aggregate[name]["n"] += 1
                aggregate[name]["loss_sum"] += loss
                by_oov[cat][name]["n"] += 1
                by_oov[cat][name]["loss_sum"] += loss
                rec["component_loss_sums"][name] += loss
        records.append(rec)

    def finalize(table):
        return {name: None if value["n"] == 0
                else round(value["loss_sum"] / value["n"], 9)
                for name, value in table.items()}

    return {
        "all_nll": finalize(aggregate),
        "oov_nll": finalize(by_oov["oov"]),
        "seen_nll": finalize(by_oov["seen"]),
        "scored_targets": aggregate["count"]["n"],
        "oov_targets": by_oov["oov"]["count"]["n"],
        "per_story": records,
    }


def paired_story_bootstrap(records, seed=BOOTSTRAP_SEED, repeats=BOOTSTRAP_REPEATS):
    """Descriptive paired 95% bootstrap interval over story units, not bytes."""
    if len(records) < 2 or repeats < 50:
        raise ValueError("insufficient story-level resampling")
    rng = random.Random(seed)
    n = len(records)
    deltas = []
    for _ in range(repeats):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        total = sum(r["scored_positions"] for r in sample)
        if total <= 0:
            raise ValueError("no scored targets")
        diff = sum(r["component_loss_sums"]["hybrid"] -
                   r["component_loss_sums"]["count"] for r in sample)
        deltas.append(diff / total)
    deltas.sort()
    return {
        "unit": "whole story (paired resampling)",
        "seed": seed,
        "repeats": repeats,
        "hybrid_minus_count_95pct_descriptive_interval_nats_byte": [
            round(deltas[int(0.025 * repeats)], 9),
            round(deltas[min(repeats-1, int(0.975 * repeats))], 9),
        ],
    }


def evaluate():
    # First freeze calibration and hyperparameters; only THEN retrieve test rows.
    train, calibration, train_dev_hashes = h.source.load_documents()
    choices = calibrate(train, calibration)
    stories = fetch_test()
    source_sha = [sha256(s.encode("utf-8")) for s in stories]
    if set(source_sha) & set(train_dev_hashes):
        raise ValueError("test duplicates train or calibration source stories")
    test, spans = normalize_and_span(stories)
    native, native_receipt = h.native_true_probs(
        train, test, epochs=EPOCHS, full_byte_vocab=True)
    gram = h.true_probs_ngram(
        train, test, k=choices["n_gram_order"], alpha=choices["alpha"],
        full_byte_vocab=True)
    metrics = summarize_scored(train, test, spans, native, gram, choices["lambda"])
    ci = paired_story_bootstrap(metrics["per_story"])
    # Avoid storing content or internal learned weights; source hashes and
    # per-story aggregate sums are sufficient to reproduce the measurement.
    out = {
        "protocol": PROTOCOL,
        "precommitted_test_split": TEST_SPLIT,
        "precommitted_offset": TEST_OFFSET,
        "precommitted_story_count": TEST_COUNT,
        "epochs": EPOCHS,
        "seed_native": 2463534242,
        "calibration_train_rows": "train 0..11",
        "calibration_dev_rows": "validation 0..5",
        "test_source_sha256": sha256(test),
        "test_source_story_sha256": source_sha,
        "train_sha256": sha256(train),
        "calibration_sha256": sha256(calibration),
        "test_bytes": len(test),
        "oov_test_byte_ids": sorted(set(test) - set(train)),
        "data_source": "https://huggingface.co/datasets/roneneldan/TinyStories",
        "dataset_license": "cdla-sharing-1.0",
        "native_runtime": "x86-64 Assembly compiled to ELF; 256-byte support, full native probability stream",
        "statistical_runtime": "Python evaluator only; count baseline and fixed mixture",
        "calibration": choices,
        "native_test_receipt_all_positions": native_receipt,
        "scoring": {
            "excludes": "first 8 targets of every story, all inter-story delimiter bytes",
            "target_scope": "only within-story positions after 8 preceding bytes",
            "nll_units": "nats/byte",
            "aggregate": {k: v for k, v in metrics.items() if k != "per_story"},
            "per_story": metrics["per_story"],
            "paired_story_bootstrap": ci,
        },
        "interpretation_limits": [
            "One fixed seed and only twelve training stories: no semantic language understanding shown",
            "Holdout row IDs frozen before fetch; data source revisions may affect reproducibility",
            "Calibration source previously inspected; hyperparameter search is small and predefined",
            "Bootstrap interval is descriptive and does not account for all experiment choices",
            "Only the native neural probability stream is in Assembly; count and blend use Python",
            "No generated-text user utility demonstrated; NLL does not measure instruction following",
            "No model checkpoint artifact is exported in incompatible full-byte v2 format",
        ],
    }
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Precommitted official TinyStories story-level OOV benchmark")
    parser.add_argument(
        "--save", type=pathlib.Path,
        default=ROOT / "evidence/preregistered-oov-story-local-20261017.json")
    args = parser.parse_args()
    result = evaluate()
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    a = result["scoring"]["aggregate"]
    print(json.dumps({
        "protocol": PROTOCOL,
        "train_sha256": result["train_sha256"],
        "test_sha256": result["test_source_sha256"],
        "story_count": TEST_COUNT,
        "scored_targets": a["scored_targets"],
        "oov_targets": a["oov_targets"],
        "all_nll": a["all_nll"],
        "oov_nll": a["oov_nll"],
        "bootstrap": result["scoring"]["paired_story_bootstrap"],
    }, indent=2))


if __name__ == "__main__":
    main()
