"""Unknown outcomes cannot become correct MAYBE answers or invented zero latency."""

from houndoracle.experiments.metrics import paired_comparison, summary


def test_failed_abstention_is_not_successful_insufficient_evidence():
    rows = [
        {
            "id": "a",
            "group_id": "a",
            "expected": "MAYBE",
            "decision": "MAYBE",
            "failure": "TIMEOUT",
            "latency_seconds": None,
        },
        {
            "id": "b",
            "group_id": "b",
            "expected": "YES",
            "decision": "YES",
            "failure": None,
            "latency_seconds": 1.0,
        },
    ]
    result = summary(rows)
    assert result["accuracy_all_decisions"] == 0.5
    assert result["failed_queries"] == 1 and result["missing_latency_count"] == 1
    assert all(v is None for v in result["latency_seconds"].values())
    assert result["all_in_cost_usd"] is None
    comparison = paired_comparison(rows, rows, replicates=20)
    assert (
        comparison["p50_latency_reduction"] is None and comparison["p95_latency_reduction"] is None
    )
