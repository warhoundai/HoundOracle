"""Development-only verdict-conditional correctness calibration; no risk/test access."""

import json
from pathlib import Path

from houndoracle.experiments.fit_calibration import fit, predict
from houndoracle.experiments.head import infer_head
from houndoracle.experiments.metrics import summary


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    public = Path("research/experiments/campaign-17")
    target = public / "screen.json"
    if target.exists():
        raise FileExistsError("Do not retune completed screen")
    head = json.loads(
        Path("research/experiments/campaign-09/continued-mixture-head.json").read_text()
    )
    source = Path("/workspace/houndoracle-data/campaign-09/scores/continued")
    rows = {
        split: [
            infer_head(r, head, evidence_group_count=r["evidence_group_count"])
            for r in load(source / (split + ".jsonl"))
        ]
        for split in ["calibration-fit", "validation"]
    }
    mapping = {}
    for label in ["YES", "NO"]:
        eligible = [
            r
            for r in rows["calibration-fit"]
            if r["candidate"] == label and not r.get("truncated") and r.get("raw_score") is not None
        ]
        if len(eligible) < 30:
            raise ValueError("Too few per-verdict calibration examples")
        mapping[label] = fit("logistic", eligible)
    baseline_mapping = json.loads(
        Path("research/experiments/campaign-09/continued-mixture-calibration.json").read_text()
    )["mapping"]
    outputs = {}
    for name, conditional in [("baseline", False), ("conditional", True)]:
        scored = []
        for row in rows["validation"]:
            probability = (
                predict(
                    mapping[row["candidate"]] if conditional else baseline_mapping, row["raw_score"]
                )
                if row["candidate"] in ["YES", "NO"]
                else None
            )
            decision = (
                row["candidate"]
                if probability is not None and probability >= 0.8 and not row.get("truncated")
                else "MAYBE"
            )
            scored.append(
                row
                | {"decision": decision, "mapped_correctness": probability, "latency_seconds": None}
            )
        outputs[name] = summary(scored)
    a, b = outputs["baseline"], outputs["conditional"]
    delta = b["correct_definitive_fraction"] - a["correct_definitive_fraction"]
    passed = delta >= 0.02 and b["definitive_coverage"] >= a["definitive_coverage"] - 0.02
    result = {
        "summaries": outputs,
        "correct_definitive_delta": delta,
        "screen_passed": passed,
        "risk_qualified": False,
        "locked_scored": False,
        "production_qualified": False,
        "decision": "qualify-fresh-risk" if passed else "reject-before-fresh-risk",
        "scope": "Reused development only; no risk certificate and no operational adoption.",
    }
    (public / "conditional-map.json").write_text(json.dumps(mapping, indent=2) + "\n")
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
