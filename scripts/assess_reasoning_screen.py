"""Reconcile every planned paired Gemini diagnostic slot before computing metrics."""

import json
from pathlib import Path

from houndoracle.experiments.metrics import paired_comparison, summary


def main():
    public = Path("research/experiments/campaign-33")
    if (public / "screen.json").exists():
        raise FileExistsError("Preserve completed screen")
    root = Path("/workspace/houndoracle-data/campaign-33")
    logs = json.loads((root / "logs.json").read_text())
    events = [
        x["jsonPayload"]
        for x in logs
        if x.get("jsonPayload", {}).get("campaign") == "houndoracle-reasoning-33"
    ]
    rows = [x["record"] for x in events if x["kind"] == "query"]
    planned = json.loads((root / "payload.json").read_text())["questions"]
    expected = {(r["id"], level): r for r in planned for level in ["LOW", "HIGH"]}
    keys = [(r["id"], r["thinking_level"]) for r in rows]
    if (
        len(set(keys)) != len(keys)
        or set(keys) != set(expected)
        or not any(x["kind"] == "completed" for x in events)
    ):
        raise ValueError("Missing/duplicate/uncertain slots; reconcile rather than hide them")
    for r in rows:
        p = expected[r["id"], r["thinking_level"]]
        if (r["group_id"], r["expected"]) != (p["group_id"], p["expected"]):
            raise ValueError("Unpaired labels")
        r["citation_supported"] = None
    low = sorted([r for r in rows if r["thinking_level"] == "LOW"], key=lambda r: r["id"])
    high = sorted([r for r in rows if r["thinking_level"] == "HIGH"], key=lambda r: r["id"])
    inputs = json.loads(Path("/workspace/houndoracle-data/campaign-32/inputs.json").read_text())
    by = {r["id"]: r for r in inputs["baseline"]}
    baseline = [
        by[r["id"]] | {"decision": by[r["id"]]["candidate"], "latency_seconds": None} for r in low
    ]
    report = {
        "systems": {
            "NLI29_raw": summary(baseline),
            "Gemini_LOW": summary(low),
            "Gemini_HIGH": summary(high),
        },
        "paired_high_vs_low": paired_comparison(low, high, seed=1970, replicates=2000),
        "reported_token_list_price_usd": {
            k: sum(
                r["token_cost_at_list_price_usd"]
                for r in rows
                if r["thinking_level"] == k and r["token_cost_at_list_price_usd"] is not None
            )
            for k in ["LOW", "HIGH"]
        },
        "planned_slots": 48,
        "observed_slots": len(rows),
        "requests_attempted": sum(r["request_attempted"] for r in rows),
        "failed_slots": sum(bool(r["failure"]) for r in rows),
        "reservations_logged": sum(x["kind"] == "reservation" for x in events),
        "all_in_cost_measured": False,
        "production_qualified": False,
        "scope": "24balanced reused development questions, one paired call per level, randomized order. Thinking-level diagnostic only; no independent99/Gemini-superiority/citation-quality/full-agent claim.",
        "records": rows,
    }
    (public / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
