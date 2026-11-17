"""Verdict agreement must not masquerade as independently reviewed citation support."""

from houndoracle.experiments.metrics import summary


def row(id, supported=True, method=None):
    return {
        "id": id,
        "group_id": id,
        "expected": "YES",
        "decision": "YES",
        "latency_seconds": 0.1,
        "citation_supported": supported,
        "citation_audit_method": method,
    }


def test_legacy_label_proxy_and_mixed_missing_audits_stay_unmeasured():
    assert summary([row("a")])["citation_supported_definitive_fraction"] is None
    assert (
        summary(
            [
                row(
                    "a",
                    method="original complete gold evidence set proxy; not a second human citation audit",
                )
            ]
        )["citation_supported_definitive_fraction"]
        is None
    )
    assert (
        summary([row("a", method="independent_span_review"), row("b")])[
            "citation_supported_definitive_fraction"
        ]
        is None
    )


def test_recorded_independent_review_can_include_failed_citations():
    result = summary(
        [row("a", method="independent_span_review"), row("b", False, "independent_span_review")]
    )
    assert result["citation_supported_definitive_fraction"] == 0.5
    assert result["accuracy_all_decisions"] == 1
