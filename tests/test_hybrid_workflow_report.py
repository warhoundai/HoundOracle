import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "hybrid_report", Path(__file__).parents[1] / "scripts/report_hybrid_workflow.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_missing_workflow_slots_and_failed_abstentions_are_not_successes():
    row = {"id": "q", "group_id": "g", "expected": "MAYBE"}
    payload = {"questions": [row], "repetitions": 3}
    event = {
        "jsonPayload": {
            "campaign": "houndoracle-workflow-18",
            "kind": "query",
            "record": row
            | {
                "repeat": 0,
                "arm": "gemini",
                "state": "FAILED",
                "executed": True,
                "workflow_latency_seconds": 60,
                "result": {"decision": "MAYBE", "failure": "TimeoutError"},
            },
        }
    }
    records, result = module.report(payload, [event], replicates=10)
    assert len(records) == 6
    assert result["summaries"]["gemini"]["accuracy_all_decisions"] == 0
    assert result["summaries"]["gemini"]["latency_seconds"]["p50"] is None
    assert result["model_token_costs"]["gemini"]["unknown_cost_slots"] == 3
    assert result["paired_cluster_bootstrap"]["p50_latency_reduction"] is None
    assert result["observed_token_cost_reduction"] is None
