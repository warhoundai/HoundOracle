"""Development replay of deterministic verdict preservation, not new inference timing."""

import json
from pathlib import Path

from houndoracle.experiments.draft_relation import preserve_or_fallback
from houndoracle.experiments.metrics import paired_comparison, summary


def main():
    public = Path("research/experiments/campaign-41")
    if (public / "screen.json").exists():
        raise FileExistsError("Preserve development replay")
    private = Path("/workspace/houndoracle-data/campaign-40")
    data = json.loads((private / "result.json").read_text())
    inputs = json.loads((private / "inputs.json").read_text())
    by = {r["id"]: r for r in inputs["questions"]}
    baseline = sorted(
        [r for r in data["records"] if r["mode"] == "reasoning-first"], key=lambda r: r["id"]
    )
    candidate = []
    overrides = []
    potential_seconds = 0
    for old in baseline:
        row = by[old["id"]]
        path = private / (old["id"].replace(":", "_") + "-draft.txt")
        new = preserve_or_fallback(path.read_text(), row["question"], len(row["evidence"]), old)
        result = (
            {k: old[k] for k in ["id", "group_id", "expected"]}
            | new
            | {
                "decision": new["candidate"],
                "latency_seconds": None,
                "citation_supported": None,
                "original_candidate": old["candidate"],
            }
        )
        candidate.append(result)
        if new["candidate"] != old["candidate"]:
            overrides.append(
                {
                    "id": old["id"],
                    "from": old["candidate"],
                    "to": new["candidate"],
                    "status": new["normalizer_status"],
                }
            )
        if new["normalizer_status"] == "EXPLICIT_DRAFT_RELATION":
            potential_seconds += sum(c["seconds"] for c in old["calls"] if c["phase"] == "final")
    reference = [r | {"decision": r["candidate"], "latency_seconds": None} for r in baseline]
    a, b = summary(reference), summary(candidate)
    passed = (
        b["accuracy_all_decisions"] >= a["accuracy_all_decisions"] + 0.02
        and b["correct_definitive_fraction"] >= a["correct_definitive_fraction"]
        and b["definitive_coverage"] >= 0.35
    )
    report = {
        "baseline_reasoning_first": a,
        "candidate_replay": b,
        "paired_comparison": paired_comparison(reference, candidate, seed=1989, replicates=2000),
        "screen_passed": passed,
        "decision": "investigate-on-fresh-qualification-only" if passed else "do-not-promote",
        "verdict_changes": overrides,
        "explicit_normalized_count": sum(
            r["normalizer_status"] == "EXPLICIT_DRAFT_RELATION" for r in candidate
        ),
        "formatter_seconds_observed_on_normalizable_cases": potential_seconds,
        "latency_note": "Counterfactual call-avoidance opportunity only; no new cold workflow latency or economics measurement.",
        "design": "Post-hoc decoder developed after3development error examples; all12replayed, no labels changed or cases dropped. Not independent.",
        "new_model_calls": 0,
        "new_cloud_spend_usd": 0,
        "production_qualified": False,
        "strict99_qualified": False,
        "fresh_risk_opened": False,
        "locked_opened": False,
        "records": candidate,
    }
    (public / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
