"""Frozen development gate for two controlled adapter-training arms."""

import json
from pathlib import Path

from houndoracle.experiments.metrics import paired_comparison, summary


def raw(rows):
    return [r | {"decision": r["candidate"], "latency_seconds": None} for r in rows]


def main():
    public = Path("research/experiments/campaign-35")
    if (public / "screen.json").exists():
        raise FileExistsError("Preserve completed screen")
    baseline = json.loads(Path("/workspace/houndoracle-data/campaign-29/result.json").read_text())[
        "scores"
    ]["validation"]
    a = summary(raw(baseline))
    reports = {}
    scores = {}
    passing = []
    for arm in ["ce-control", "contrastive"]:
        record = json.loads(
            (Path("/workspace/houndoracle-data/campaign-35") / arm / "result.json").read_text()
        )
        if not record["completed"] or record["steps"] != 128:
            reports[arm] = {
                "completed": False,
                "screen_passed": False,
                "reason": "INCOMPLETE_TRAINING",
            }
            continue
        rows = record["validation_scores"]
        if [(r["id"], r["group_id"], r["expected"]) for r in rows] != [
            (r["id"], r["group_id"], r["expected"]) for r in baseline
        ]:
            raise ValueError("Unpaired development observations")
        b = summary(raw(rows))
        passed = (
            b["accuracy_all_decisions"] >= a["accuracy_all_decisions"] + 0.02
            and b["definitive_coverage"] >= a["definitive_coverage"] - 0.02
        )
        scores[arm] = rows
        reports[arm] = {
            "completed": True,
            "training": {
                k: v for k, v in record.items() if k not in ["validation_scores", "losses"]
            },
            "raw": b,
            "paired_vs_base": paired_comparison(
                raw(baseline), raw(rows), seed=1983, replicates=2000
            ),
            "screen_passed": passed,
        }
        if passed:
            passing.append(arm)
    selected = (
        max(
            passing,
            key=lambda n: (
                reports[n]["raw"]["correct_definitive_fraction"],
                reports[n]["raw"]["accuracy_all_decisions"],
                n == "ce-control",
            ),
        )
        if passing
        else None
    )
    result = {
        "baseline_raw": a,
        "arms": reports,
        "selected": selected,
        "decision": "retain-selected-for-fresh-qualification"
        if selected
        else "reject-both-before-new-held-out-data",
        "production_qualified": False,
        "fresh_risk_opened": False,
        "locked_opened": False,
        "calibration_fitted": False,
        "new_cloud_spend_usd": 0,
        "scope": "Reused207development queries only; adapter losses or softmax scores are not factual confidence. Base29retained unchanged.",
        "records": scores,
    }
    if len(scores) == 2:
        result["paired_contrastive_vs_ce"] = paired_comparison(
            raw(scores["ce-control"]), raw(scores["contrastive"]), seed=1983, replicates=2000
        )
    (public / "screen.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
