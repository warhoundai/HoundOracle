"""Reconcile every preregistered slot before reporting the cloud policy pilot."""

import hashlib
import json
from pathlib import Path

from scipy.stats import beta

from houndoracle.experiments.metrics import paired_comparison, summary


def main():
    public = Path("research/experiments/campaign-42")
    private = Path("/workspace/houndoracle-data/campaign-42")
    if (public / "screen.json").exists():
        raise FileExistsError("Completed results must be preserved")
    logs = json.loads((private / "logs.json").read_text())
    events = [r["jsonPayload"] for r in logs if "jsonPayload" in r]
    complete = [e for e in events if e.get("kind") == "completed"]
    if len(complete) != 1:
        raise ValueError("Wait for completion; never report a successful prefix")
    payload = json.loads((private / "payload.json").read_text())
    protocol = json.loads((public / "protocol.json").read_text())
    digest = hashlib.sha256(json.dumps(payload["questions"], sort_keys=True).encode()).hexdigest()
    configurations = [e for e in events if e.get("kind") == "configuration"]
    if (
        len(configurations) != 1
        or configurations[0]["dataset_sha256"] != digest
        or protocol["dataset_sha256"] != digest
        or any(c["thinking_level"] != "HIGH" for c in configurations[0]["models"].values())
    ):
        raise ValueError("Dataset or model configuration changed")
    rows = [e["record"] for e in events if e.get("kind") == "query"]
    expected = {
        (r["id"], policy): r
        for r in payload["questions"]
        for policy in ["baseline", "relation_audit"]
    }
    keys = [(r["id"], r["thinking_level"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError("Missing/duplicate experimental slots")
    for row in rows:
        original = expected[row["id"], row["thinking_level"]]
        assert (row["group_id"], row["expected"]) == (original["group_id"], original["expected"])
    arms = {
        policy: sorted(
            [
                r | {"policy": policy, "thinking_level": "HIGH"}
                for r in rows
                if r["thinking_level"] == policy
            ],
            key=lambda r: r["id"],
        )
        for policy in ["baseline", "relation_audit"]
    }
    stats = {k: summary(v) for k, v in arms.items()}
    for policy, values in arms.items():
        k = sum(not r.get("failure") and r["decision"] == r["expected"] for r in values)
        n = len(values)
        stats[policy]["accuracy_exact_95pct_interval"] = [
            float(beta.ppf(0.025, k, n - k + 1)) if k else 0,
            float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1,
        ]
        known = [
            r["token_cost_at_list_price_usd"]
            for r in values
            if r.get("token_cost_at_list_price_usd") is not None
        ]
        stats[policy]["known_token_list_price_subtotal_usd"] = sum(known)
        stats[policy]["token_cost_observed_count"] = len(known)
        stats[policy]["reported_token_list_price_usd"] = sum(known) if len(known) == n else None
    historical = json.loads(Path("/workspace/houndoracle-data/campaign-33/inputs.json").read_text())
    old = {r["id"]: r for r in historical["baseline"]}
    reference = []
    for row in arms["baseline"]:
        previous = old[row["id"]]
        assert previous["expected"] == row["expected"]
        reference.append(previous | {"decision": previous["candidate"], "latency_seconds": None})
    a, b = stats["baseline"], stats["relation_audit"]
    passed = (
        not any(r.get("failure") for r in rows)
        and b["accuracy_all_decisions"] >= a["accuracy_all_decisions"] + 2 / 24 - 1e-12
        and b["correct_definitive_count"] >= a["correct_definitive_count"]
    )
    result = {
        "systems": stats,
        "historical_local_NLI29_reference": summary(reference),
        "local_reference_note": "Previously scored identical supplied evidence cases, no new local timing or agent workflow measurement. Raw uncalibrated component scores.",
        "label_breakdown": {
            policy: {
                label: summary([r for r in values if r["expected"] == label])
                for label in ["YES", "NO", "MAYBE"]
            }
            for policy, values in arms.items()
        },
        "paired": paired_comparison(
            arms["baseline"], arms["relation_audit"], seed=1991, replicates=2000
        ),
        "all_planned_slots": len(rows),
        "screen_passed": passed,
        "observed_95pct_target_met": b["accuracy_all_decisions"] >= 0.95,
        "independently_qualified_95pct": False,
        "production_qualified": False,
        "scope": "Reused24development cases, matched supplied historical evidence; no live search or whole-agent comparison. Citation entailment unaudited; model candidates uncalibrated. No label changes or failed-row exclusions.",
        "completion": complete[0],
    }
    (public / "screen.json").write_text(json.dumps(result, indent=2) + "\n")
    output = Path("benchmarks/results/campaign-42")
    output.mkdir(parents=True, exist_ok=True)
    (output / "development-metadata.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for rs in arms.values() for r in rs)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
