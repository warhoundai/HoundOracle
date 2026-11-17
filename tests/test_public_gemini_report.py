import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "pilot_report", Path(__file__).parents[1] / "scripts/report_public_gemini_pilot.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_missing_failed_and_duplicate_logs_remain_accounted():
    row = {"id": "q", "group_id": "g", "expected": "MAYBE"}
    payload = {"questions": [row], "repetitions": 3}
    envelope = row | {
        "repeat": 0,
        "state": "FAILED",
        "result": {"decision": "MAYBE", "failure": "TimeoutError", "latency_seconds": 60},
    }
    entry = {
        "jsonPayload": {
            "campaign": module.CAMPAIGN,
            "kind": "query",
            "record": envelope,
        }
    }
    local = [
        row | {"repeat": i, "candidate": "MAYBE", "decision": "MAYBE", "latency_seconds": 0.05}
        for i in range(3)
    ]
    records, report = module.report(payload, [entry, entry], local, replicates=10)
    assert len(records) == 3
    assert report["gemini_matched_evidence"]["accuracy_all_decisions"] == 0
    assert report["gemini_matched_evidence"]["failed_queries"] == 3
    assert report["gemini_matched_evidence"]["latency_seconds"]["p50"] is None
    assert report["gemini_token_list_price_usd_bounds"] == [0, 3 * module.MAXIMUM_RESERVATION]
    assert report["local_minus_gemini_paired_cluster_bootstrap"]["p50_latency_reduction"] is None
    conflict = {"jsonPayload": entry["jsonPayload"] | {"record": envelope | {"state": "SUCCEEDED"}}}
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        module.reconcile(payload, [entry, conflict])


def test_public_trace_keeps_provenance_without_redistributing_source_passage():
    row = {"id": "q", "group_id": "g", "expected": "YES"}
    payload = {"questions": [row], "repetitions": 1}
    event = {
        "jsonPayload": {
            "campaign": module.CAMPAIGN,
            "kind": "query",
            "record": row
            | {
                "repeat": 0,
                "state": "SUCCEEDED",
                "result": {
                    "decision": "YES",
                    "answer": {
                        "decision": "YES",
                        "evidence_indices": [0],
                        "citations": [
                            {"url": "https://example.org", "quote": "A licensed evidence passage."}
                        ],
                    },
                },
            },
        }
    }
    records, _ = module.reconcile(payload, [event])
    assert "quote" not in records[0]["answer"].get("citation_metadata", [{}])[0]
    assert records[0]["answer"]["citation_metadata"][0]["quote_characters"] == len(
        "A licensed evidence passage."
    )
    assert len(records[0]["answer"]["citation_metadata"][0]["quote_sha256"]) == 64
