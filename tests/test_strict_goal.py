from houndoracle.experiments.strict_goal import evaluate_goal


def cohort():
    return [
        {
            "id": str(i),
            "group_id": str(i),
            "expected": "YES" if i < 400 else "MAYBE",
            "decision": "YES" if i < 400 else "MAYBE",
            "confidence": 1.0,
            "citation_supported": True,
            "citation_audit_method": "independent_span_review",
        }
        for i in range(600)
    ]


def test_perfect_independent_audited_cohort_passes_both_bounds():
    report = evaluate_goal(cohort())
    assert report["gate_passed"] and report["accuracy_lower_bound"] >= 0.99


def test_high_conditional_precision_cannot_hide_incorrect_abstentions():
    rows = cohort()
    for row in rows[400:]:
        row["expected"] = "YES"
    report = evaluate_goal(rows)
    assert report["selective"]["empirical_precision"] == 1
    assert report["selective"]["gate_passed"] and not report["gate_passed"]


def test_missing_audit_and_repeated_network_failure_cannot_qualify():
    rows = cohort()
    rows[0]["citation_supported"] = None
    assert not evaluate_goal(rows)["gate_passed"]
    rows = cohort()
    for row in rows:
        row["failure"] = "NETWORK_ERROR"
    assert not evaluate_goal(rows)["gate_passed"]
