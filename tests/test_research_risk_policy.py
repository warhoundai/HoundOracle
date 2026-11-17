"""Per-query exact bounds and legacy cluster bounds must both enforce the confidence floor."""

import pytest

from houndoracle.experiments.pipeline import apply


def artifact(certificate):
    return {
        "mapping": {"method": "logistic", "weight": 1.0, "bias": 0.0},
        "certificate": certificate,
    }


@pytest.mark.parametrize("upper", [None, float("nan"), float("inf"), -0.1, 1.1, True, 0.21])
def test_invalid_or_insufficient_risk_bounds_cannot_emit_definitive_answers(upper):
    result = apply(
        {"candidate": "YES", "raw_score": 0.99},
        artifact({"qualified": True, "row_risk_upper": upper}),
    )
    assert result["decision"] == "MAYBE" and result["confidence"] is None


def test_exact_query_risk_bound_and_legacy_certificate_remain_supported():
    row = {"candidate": "NO", "raw_score": 0.99}
    for key in ["row_risk_upper", "row_risk_upper_bootstrap"]:
        result = apply(row, artifact({"qualified": True, key: 0.1}))
        assert result["decision"] == "NO" and result["confidence"] == 0.9
    assert (
        apply(row, artifact({"qualified": "false", "row_risk_upper": 0.01}))["decision"] == "MAYBE"
    )
    assert (
        apply(row | {"truncated": True}, artifact({"qualified": True, "row_risk_upper": 0.01}))[
            "decision"
        ]
        == "MAYBE"
    )
