#!/usr/bin/env python3
"""Offline invariants for preregistered source selection and native-score isolation."""
import hashlib
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import preregistered_eval as p


def rejects(fn, contains=None):
    try:
        fn()
    except ValueError as ex:
        if contains:
            assert contains in str(ex), (contains, ex)
    else:
        raise AssertionError("expected fail-closed rejection")


def source_contract():
    assert p.TEST_OFFSET == 1200 and p.TEST_COUNT == 24
    rows = [{"row_idx": i, "row": {"text": "source story unique " + str(i)}}
            for i in range(p.TEST_OFFSET, p.TEST_OFFSET+p.TEST_COUNT)]
    assert p.validated_rows({"rows": rows})[0].endswith("1200")
    rows[0]["row_idx"] = p.TEST_OFFSET + 1
    rejects(lambda: p.validated_rows({"rows": rows}), "unexpected test rows")
    rows[0]["row_idx"] = p.TEST_OFFSET
    rows[0]["row"]["text"] = rows[1]["row"]["text"]
    rejects(lambda: p.validated_rows({"rows": rows}), "duplicate")
    rows[0]["row"]["text"] = "source story unique 1200"
    rows[0]["row"]["text"] = None
    rejects(lambda: p.validated_rows({"rows": rows}), "missing text")
    print("PASS: locked source IDs, duplicate source and malformed content reject")


def score_contract():
    stories = ["the cat sat on the mat.", "the mouse sees 0123.", "and the dog ran home!"]
    test, spans = p.normalize_and_span(stories)
    assert len(spans) == 3
    assert test[spans[0][1]:spans[1][0]] == b"\n\n"
    assert test[spans[1][1]:spans[2][0]] == b"\n\n"
    indices = p.story_indices(spans)
    assert all(v[0] == start + p.CONTEXT_BYTES
               for v, (start, end) in zip(indices, spans))
    assert all(v[-1] == end-1 for v, (start, end) in zip(indices, spans))
    assert sum(map(len, indices)) == sum(end-start-p.CONTEXT_BYTES
                                         for start, end in spans)
    n = len(test)-p.CONTEXT_BYTES
    gram = [0.125]*n
    neural = [0.25]*n
    train = b"the cat sat and the dog mouse sees on mat ran home!"
    report = p.summarize_scored(train, test, spans, neural, gram, 0.4)
    total = sum(len(x) for x in indices)
    assert report["scored_targets"] == total
    assert report["oov_targets"] == 6   # 4 digits plus 2 previously unseen periods
    assert abs(report["all_nll"]["count"] - math.log(8)) < 1e-8
    assert abs(report["all_nll"]["native"] - math.log(4)) < 1e-8
    assert abs(report["all_nll"]["hybrid"] + math.log(.175)) < 1e-8
    assert report["oov_nll"]["hybrid"] is not None
    # Falsify probabilities for the non-scored first-eight/delimiter regions.
    gram2, neural2 = gram.copy(), neural.copy()
    scored = {pos-p.CONTEXT_BYTES for row in indices for pos in row}
    for j in range(n):
        if j not in scored:
            gram2[j] = 1e-20
            neural2[j] = 1e-15
    report2 = p.summarize_scored(train, test, spans, neural2, gram2, 0.4)
    assert report["all_nll"] == report2["all_nll"]
    assert report["oov_nll"] == report2["oov_nll"]
    assert report["seen_nll"] == report2["seen_nll"]
    rejects(lambda: p.summarize_scored(train, test, spans, neural[:-1], gram, 0.4),
            "wrong length")
    gram2 = gram.copy()
    gram2[indices[0][0]-p.CONTEXT_BYTES] = float("nan")
    rejects(lambda: p.summarize_scored(train, test, spans, neural, gram2, 0.4),
            "invalid probability")
    print("PASS: story-boundary exclusion and OOV byte-level partitions")


def calibration_contract():
    train = b"the little mouse ate the cheese.\n"*30
    calibration = b"the mouse ate all the cheese.\n"*3
    old_train, old_cal = p.EXPECTED_TRAIN_SHA256, p.EXPECTED_CAL_SHA256
    try:
        p.EXPECTED_TRAIN_SHA256 = hashlib.sha256(train).hexdigest()
        p.EXPECTED_CAL_SHA256 = hashlib.sha256(calibration).hexdigest()
        fake_calls = []
        def predictor(t, c, *, epochs, full_byte_vocab):
            fake_calls.append((t, c, epochs, full_byte_vocab))
            return [0.1]*(len(c)-p.CONTEXT_BYTES), {"synthetic": True}
        choice = p.calibrate(train, calibration, native_predictor=predictor)
        assert fake_calls == [(train,calibration,p.EPOCHS,True)]
        assert choice["n_gram_order"] in p.NGRAM_ORDERS
        assert choice["alpha"] in p.ALPHAS
        assert choice["lambda"] in p.LAMBDAS
        assert len(choice["count_calibration_grid"]) == len(p.ALPHAS)*len(p.NGRAM_ORDERS)
        rejects(lambda: p.calibrate(train+b"tamper", calibration, native_predictor=predictor),
                "input changed")
    finally:
        p.EXPECTED_TRAIN_SHA256,p.EXPECTED_CAL_SHA256 = old_train,old_cal
    print("PASS: only calibration corpus controls model-selection grids")


def confidence_contract():
    _, spans = p.normalize_and_span(["some length source story one" ,
                                     "different length source story two",
                                     "yet another long source story three"])
    n = [end-start-p.CONTEXT_BYTES for start,end in spans]
    rec = [{"scored_positions":v,
            "component_loss_sums":{"count":v*2.0,"hybrid":v*1.5}}
           for v in n]
    a = p.paired_story_bootstrap(rec, repeats=200)
    b = p.paired_story_bootstrap(rec, repeats=200)
    assert a == b
    assert a["hybrid_minus_count_95pct_descriptive_interval_nats_byte"] == [-0.5,-0.5]
    rejects(lambda: p.paired_story_bootstrap(rec[:1]), "insufficient")
    print("PASS: story-level paired resampling is deterministic; no byte pseudoreplication")


if __name__ == "__main__":
    source_contract()
    score_contract()
    calibration_contract()
    confidence_contract()
    print("PASS: independent offline preregistered evaluation gates")
