"""Evaluate completed adaptation on development only; preserve independent risk/test."""

import json
import math
from pathlib import Path

from houndoracle.experiments.fit_calibration import fit, predict
from houndoracle.experiments.head import infer_head
from houndoracle.experiments.metrics import paired_comparison, summary

PUBLIC = Path("research/experiments/campaign-29")
PRIVATE = Path("/workspace/houndoracle-data/campaign-29")


def load(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def select(rows, mapping, threshold):
    return [
        r
        | {
            "decision": r["candidate"]
            if r["candidate"] in ["YES", "NO"]
            and not r.get("truncated")
            and predict(mapping, r["raw_score"]) >= threshold
            else "MAYBE"
        }
        for r in rows
    ]


def main():
    out = PUBLIC / "screen.json"
    if out.exists():
        raise FileExistsError("Preserve completed candidate assessment")
    data = json.loads((PRIVATE / "result.json").read_text())
    training = data["training"]
    if (
        not training["completed"]
        or training["steps"] != 800
        or not all(math.isfinite(v) for v in training["losses"])
    ):
        report = {
            "screen_passed": False,
            "decision": "reject-incomplete-or-invalid-training",
            "training": training,
            "fresh_risk_opened": False,
            "locked_opened": False,
        }
    else:
        val = data["scores"]["validation"]
        cal = data["scores"]["calibration-fit"]
        if {r["group_id"] for r in val} & {r["group_id"] for r in cal}:
            raise ValueError("Split leakage")
        head = json.loads(
            Path("research/experiments/campaign-09/continued-mixture-head.json").read_text()
        )
        oldmap = json.loads(
            Path("research/experiments/campaign-09/continued-mixture-calibration.json").read_text()
        )["mapping"]
        baseline = [
            infer_head(r, head, evidence_group_count=r["evidence_group_count"])
            for r in load(
                "/workspace/houndoracle-data/campaign-09/scores/continued/validation.jsonl"
            )
        ]
        if [(r["id"], r["group_id"], r["expected"]) for r in val] != [
            (r["id"], r["group_id"], r["expected"]) for r in baseline
        ]:
            raise ValueError("Unpaired validation")
        mapping = fit("logistic", [r for r in cal if not r.get("truncated")])
        raw = lambda rows: [r | {"decision": r["candidate"], "latency_seconds": None} for r in rows]
        a, b = summary(raw(baseline)), summary(raw(val))
        ac, bc = summary(select(baseline, oldmap, 0.8)), summary(select(val, mapping, 0.8))
        strict = summary(select(val, mapping, 0.99))
        delta = b["accuracy_all_decisions"] - a["accuracy_all_decisions"]
        gain = bc["correct_definitive_fraction"] - ac["correct_definitive_fraction"]
        passed = (
            (delta >= 0.02 or gain >= 0.02)
            and delta >= -0.01
            and bc["definitive_coverage"] >= ac["definitive_coverage"] - 0.02
        )
        report = {
            "training": {k: v for k, v in training.items() if k != "losses"},
            "baseline_raw": a,
            "candidate_raw": b,
            "baseline_provisional80": ac,
            "candidate_provisional80": bc,
            "candidate_provisional99": strict,
            "raw_accuracy_delta": delta,
            "correct_definitive_delta": gain,
            "paired_raw_comparison": paired_comparison(
                raw(baseline), raw(val), seed=1963, replicates=2000
            ),
            "screen_passed": bool(passed),
            "target99_plausible_on_development": bool(
                strict["accuracy_all_decisions"] >= 0.99
                and strict["definitive_coverage"] >= 0.35
                and strict["selective_risk"] is not None
                and strict["selective_risk"] <= 0.01
            ),
            "decision": "retain-research-candidate-only" if passed else "reject-before-fresh-risk",
            "fresh_risk_opened": False,
            "locked_opened": False,
            "production_qualified": False,
            "scope": "Reused development; fitted correctness map unqualified, citation audit absent. Host timings not paired.99% all-decision/precision/coverage target not established.",
        }
        (PUBLIC / "provisional-mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
