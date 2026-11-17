"""Reconcile paired development slots; compatibility failures are not model-quality results."""

import json
from pathlib import Path

from houndoracle.experiments.metrics import paired_comparison, summary


def main():
    public = Path("research/experiments/campaign-40")
    private = Path("/workspace/houndoracle-data/campaign-40")
    if (public / "screen.json").exists():
        raise FileExistsError("Preserve completed screen")
    data = json.loads((private / "result.json").read_text())
    inputs = json.loads((private / "inputs.json").read_text())
    rows = data["records"]
    expected = {
        (r["id"], mode): r for r in inputs["questions"] for mode in ["direct", "reasoning-first"]
    }
    keys = [(r["id"], r["mode"]) for r in rows]
    if not data["completed"] or len(set(keys)) != len(keys) or set(keys) != set(expected):
        raise ValueError("Missing or duplicate planned slots; reconcile first")
    for r in rows:
        e = expected[r["id"], r["mode"]]
        if (r["group_id"], r["expected"]) != (e["group_id"], e["expected"]):
            raise ValueError("Unpaired source labels")
    modes = {
        mode: sorted(
            [
                r | {"decision": r["candidate"], "citation_supported": None}
                for r in rows
                if r["mode"] == mode
            ],
            key=lambda r: r["id"],
        )
        for mode in ["direct", "reasoning-first"]
    }
    by = {r["id"]: r for r in inputs["baseline"]}
    baseline = [
        by[r["id"]] | {"decision": by[r["id"]]["candidate"], "latency_seconds": None}
        for r in modes["direct"]
    ]
    stats = {mode: summary(rs) for mode, rs in modes.items()}
    valid = all(not r["failure"] for r in rows)
    a, b = stats["direct"], stats["reasoning-first"]
    passed = (
        valid
        and b["accuracy_all_decisions"] >= a["accuracy_all_decisions"] + 2 / 12 - 1e-12
        and b["correct_definitive_fraction"] >= a["correct_definitive_fraction"]
        and b["definitive_coverage"] >= 0.35
    )
    report = {
        "systems": {"NLI29_raw": summary(baseline), **stats},
        "paired_reasoning_vs_direct": paired_comparison(
            modes["direct"], modes["reasoning-first"], seed=1989, replicates=2000
        ),
        "all_planned_slots": len(rows),
        "failed_slots": sum(bool(r["failure"]) for r in rows),
        "reasoning_truncated_slots": sum(bool(r.get("reasoning_truncated")) for r in rows),
        "screen_passed": passed,
        "decision": "retain-for-independent-qualification"
        if passed
        else (
            "inconclusive-runtime-or-budget-failure"
            if not valid
            else "declared-reasoning-gain-not-established"
        ),
        "resources": {
            k: data[k]
            for k in ["acquisition_seconds", "total_seconds", "peak_rss_kib", "model_calls"]
        },
        "additional_unscored_readiness_calls": data.get("additional_unscored_readiness_calls", 0),
        "runtime": data["runtime"],
        "new_cloud_spend_usd": 0,
        "production_qualified": False,
        "strict99_qualified": False,
        "fresh_risk_opened": False,
        "locked_opened": False,
        "scope": "12balanced reused development groups, not24independent cases or a99certificate. Both arms same local reasoning model; analysis is not independent evidence. Source labels unchanged, citations unaudited. No all-in or general-web claim.",
        "records": rows,
    }
    (public / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
