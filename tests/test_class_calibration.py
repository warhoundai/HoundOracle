import pytest

from houndoracle.experiments.class_calibration import SCHEMA, apply


def state():
    return {
        "feature_schema": SCHEMA,
        "classes": [0, 1, 2],
        "mean": [0, 0, 0],
        "scale": [1, 1, 1],
        "weights": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "bias": [0, 0, 0],
    }


def test_class_order_preserves_yes_no_meaning_and_has_no_truth_confidence():
    r = apply({"logits": [5, -2, 0]}, state())
    assert r["candidate"] == "NO" and r["calibrated_correctness"] is None
    assert not r["production_qualified"]
    assert apply({"logits": [-2, 5, 0]}, state())["candidate"] == "YES"


def test_label_permutation_or_nonfinite_input_is_rejected():
    s = state()
    s["classes"] = [1, 0, 2]
    with pytest.raises(ValueError):
        apply({"logits": [1, 0, 0]}, s)
    with pytest.raises(ValueError):
        apply({"logits": [float("nan"), 0, 0]}, state())
    s = state()
    s["scale"][0] = 0
    with pytest.raises(ValueError):
        apply({"logits": [1, 0, 0]}, s)
