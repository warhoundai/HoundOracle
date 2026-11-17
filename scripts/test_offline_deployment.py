"""EC2 runtime smoke/regression in an OS network namespace without outbound access."""

import json
import socket
import time
from pathlib import Path

from houndoracle import Oracle
from houndoracle.experiments.head import HeadNLI
from houndoracle.experiments.nli import OnnxNLI
from houndoracle.experiments.pipeline import ResearchPipeline
from houndoracle.experiments.precision_gate import evaluate


def main():
    target = Path("/opt/houndoracle/offline-result.json")
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=0.2)
    except OSError:
        isolated = True
    else:
        raise AssertionError("Expected OS network isolation")
    payload = json.loads(Path("/opt/houndoracle/benchmark.json").read_text())
    started = time.perf_counter()
    model = ResearchPipeline(
        HeadNLI(
            OnnxNLI(
                "/opt/houndoracle/model",
                variant="fp32",
                threads=1,
                max_length=256,
                reviewed_manifest=payload["manifest"],
            ),
            payload["head"],
            "continued-mixture",
        ),
        payload["calibration"],
    )
    init = time.perf_counter() - started
    rows = []
    mismatches = []
    for source in payload["questions"]:
        result = model.verify_evidence(
            source["question"],
            source["evidence"],
            population=payload["calibration"]["population"],
            use_cache=False,
        )
        if result["decision"] != source["frozen_decision"]:
            mismatches.append(source["id"])
        rows.append(
            {
                "id": source["id"],
                "group_id": source["group_id"],
                "expected": source["expected"],
                "decision": result["decision"],
                "confidence": result.get("confidence"),
                "latency_seconds": result["latency_seconds"],
                "citation_supported": None,
            }
        )
    accepted = [r for r in rows if r["decision"] in ["YES", "NO"]]
    strict = [
        r | {"decision": r["decision"] if (r["confidence"] or 0) >= 0.99 else "MAYBE"} for r in rows
    ]
    report = {
        "os_network_isolated": isolated,
        "external_api_calls": 0,
        "credentials_required": False,
        "initialization_seconds": init,
        "questions": len(rows),
        "runtime_fidelity_mismatches": mismatches,
        "accepted": len(accepted),
        "accepted_errors": sum(r["decision"] != r["expected"] for r in accepted),
        "correct_definitive_fraction": sum(r["decision"] == r["expected"] for r in accepted)
        / len(rows),
        "high_precision_target": evaluate(strict, minimum_coverage=0.35),
        "scope": "Independent EC2 hardware regression on previously evaluated frozen600-query historical evidence cohort. Not a new independent quality measurement. Citation entailment unaudited.99% target gate can fail; do not relabel it as achieved.",
    }
    target.write_text(json.dumps(report, indent=2) + "\n")
    Path("/opt/houndoracle/offline-records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    Oracle(local_only=True).store.close()
    if mismatches:
        raise AssertionError("Frozen verdict drift on EC2")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
