from houndoracle.experiments.precision_gate import evaluate


def records(n):
    return [
        {
            "id": str(i),
            "group_id": str(i),
            "expected": "YES",
            "decision": "YES",
            "confidence": 0.99,
            "citation_supported": True,
            "citation_audit_method": "independent_span_review",
        }
        for i in range(n)
    ]


def test_high_precision_needs_independent_support_not_abstention_or_raw_confidence():
    assert evaluate(records(299))["gate_passed"]
    assert not evaluate(records(298))["gate_passed"]
    assert not evaluate([r | {"decision": "MAYBE"} for r in records(1000)])["gate_passed"]
    assert not evaluate([r | {"citation_supported": None} for r in records(1000)])["gate_passed"]
    assert not evaluate([r | {"confidence": 0.8} for r in records(1000)])["gate_passed"]
    rows = records(299)
    rows[0]["failure"] = "NETWORK_ERROR"
    assert not evaluate(rows)["gate_passed"]
