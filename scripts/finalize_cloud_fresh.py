"""Merge a bounded continuation without dropping failed/unexecuted tasks or inventing usage."""

import json
import random
from pathlib import Path


def main():
    data = Path("/workspace/houndoracle-data")
    initial = json.loads((data / "cloud-05-initial-logs.json").read_text())
    recovery = json.loads((data / "cloud-05-recovery-logs.json").read_text())
    failed = json.loads((data / "cloud-05-failure-record.json").read_text())
    failed["execution_segment"] = "initial_failure_reconstructed"
    failed["record_origin"] = (
        "Observed provider 504 traceback; exact duration and usage unavailable."
    )
    for item in initial:
        record = item.get("jsonPayload", {}).get("record")
        if record:
            record["execution_segment"] = "initial"
    combined = (
        initial
        + [
            {
                "insertId": "reconstructed-original-504",
                "jsonPayload": {"campaign": "oracle-cloud-05", "kind": "query", "record": failed},
            }
        ]
        + recovery
    )
    payload = json.loads((data / "cloud-fresh-payload.json").read_text())
    existing = {
        (
            r["jsonPayload"]["record"]["id"],
            r["jsonPayload"]["record"]["repeat"],
            r["jsonPayload"]["record"]["arm"],
        )
        for r in combined
        if r.get("jsonPayload", {}).get("kind") == "query"
    }
    plan = []
    rng = random.Random(1907)
    for repeat in range(3):
        for row in payload["confirmation"]:
            order = ["baseline", "selected"]
            rng.shuffle(order)
            plan.extend((row, repeat, arm) for arm in order)
    missing = [
        (row, repeat, arm) for row, repeat, arm in plan if (row["id"], repeat, arm) not in existing
    ]
    for i, (row, repeat, arm) in enumerate(missing):
        # A killed serial worker may have one request in flight; retain its reservation conservatively.
        uncertain = i == 0
        record = {
            "id": row["id"],
            "group_id": row["group_id"],
            "repeat": repeat,
            "arm": arm,
            "phase": "fresh",
            "expected": row["expected"],
            "decision": "MAYBE",
            "failure": "WORKER_STOP_UNCERTAIN_ATTEMPT" if uncertain else "NOT_EXECUTED",
            "source": "runtime_limit",
            "gemini": None,
            "oracle": None,
            "model_calls": 1 if uncertain else 0,
            "latency_seconds": None,
            "token_cost_usd": None if uncertain else 0,
            "reserved_usd": 0.021504 if uncertain else 0,
            "execution_segment": "reconstructed_at_resource_cutoff",
        }
        combined.append(
            {
                "insertId": f"reconstructed-missing-{i}",
                "jsonPayload": {"campaign": "oracle-cloud-05", "kind": "query", "record": record},
            }
        )
    (data / "cloud-05-combined-logs.json").write_text(json.dumps(combined, indent=2) + "\n")
    print(
        json.dumps(
            {
                "planned_records": len(plan),
                "missing_at_cutoff": len(missing),
                "recorded_events": len(combined),
                "complete": not missing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
