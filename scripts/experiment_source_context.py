"""Controlled source-context screens; old validation is development evidence, never fresh test."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from analyze_campaign import analyze
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from houndoracle.experiments.head import head_digest, infer_head
from houndoracle.experiments.metrics import summary
from houndoracle.experiments.nli import OnnxNLI
from houndoracle.experiments.pipeline import apply
from houndoracle.experiments.source_context import SCHEMA, SourceContextNLI, grounding_features


def load(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "variant", choices=["source-original", "source-finetuned", "grounding-head"]
    )
    variant = parser.parse_args().variant
    public = Path("research/experiments/campaign-07")
    private = Path("/workspace/houndoracle-data/campaign-07") / variant
    if private.exists():
        raise FileExistsError("Preserve prior experiment; cannot silently rerun/tune")
    private.mkdir(parents=True)
    start = time.perf_counter()
    fine = variant == "source-finetuned"
    manifest = (
        json.loads(Path("research/model-comparisons/finetuned-minilm-provenance.json").read_text())
        if fine
        else None
    )
    encoder = OnnxNLI(
        "/workspace/houndoracle-models/" + ("minilm-fever-1912" if fine else "minilm"),
        variant="fp32",
        threads=2,
        reviewed_manifest=manifest,
    )
    model = SourceContextNLI(encoder) if variant != "grounding-head" else encoder
    scores = {}
    for split in ["development", "validation", "calibration"]:
        rows = load(f"/workspace/houndoracle-data/campaign-02/{split}.jsonl")
        frozen = (
            {r["id"]: r for r in load(f"/workspace/houndoracle-data/scores/{split}-fp32.jsonl")}
            if variant == "grounding-head"
            else None
        )
        scores[split] = []
        for i, row in enumerate(rows):
            raw = (
                frozen[row["id"]]
                if frozen is not None
                else model.predict(row["question"], row["evidence"])
            )
            scores[split].append(
                {k: row[k] for k in ["id", "group_id", "expected", "category"]}
                | raw
                | {
                    "latency_seconds": raw.get("model_seconds", raw.get("latency_seconds")),
                    "evidence_group_count": len({e["page"] for e in row["evidence"]}),
                    "model_input_tokens": raw.get("token_count", raw.get("model_input_tokens")),
                    "grounding_features": grounding_features(row["question"], row["evidence"])
                    if variant == "grounding-head"
                    else [],
                    "citation_supported": None,
                    "all_in_cost_usd": None,
                }
            )
            if (i + 1) % 500 == 0:
                print(
                    json.dumps(
                        {
                            "variant": variant,
                            "split": split,
                            "n": i + 1,
                            "seconds": time.perf_counter() - start,
                        }
                    ),
                    flush=True,
                )

    def vector(r):
        return (
            (r["logits"] or [0, 0, 1])
            + [np.log1p(r.get("model_input_tokens") or 0), np.log1p(r["evidence_group_count"])]
            + r["grounding_features"]
        )

    x = np.array([vector(r) for r in scores["development"]])
    y = np.array([{"NO": 0, "YES": 1, "MAYBE": 2}[r["expected"]] for r in scores["development"]])
    scaler = StandardScaler().fit(x)
    head = LogisticRegression(C=1, max_iter=500, random_state=1916).fit(scaler.transform(x), y)
    checkpoint = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "weights": head.coef_.tolist(),
        "biases": head.intercept_.tolist(),
        "classes": head.classes_.tolist(),
    }
    (public / f"{variant}-head.json").write_text(json.dumps(checkpoint, indent=2) + "\n")
    config = model.config | {"aggregation_head": variant, "head_digest": head_digest(checkpoint)}
    if variant == "grounding-head":
        config["grounding_feature_schema"] = SCHEMA
    (private / f"model-{variant}.json").write_text(json.dumps(config) + "\n")
    for split, rows in scores.items():
        path = private / f"{split}-{variant}.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    infer_head(
                        r,
                        checkpoint,
                        evidence_group_count=r["evidence_group_count"],
                        extra_features=r["grounding_features"],
                    )
                )
                + "\n"
                for r in rows
            )
        )
    analyze(private, public, variant)
    calibrated = load(public / f"validation-calibrated-{variant}.jsonl")
    (public / f"validation-calibrated-{variant}.jsonl").replace(
        private / f"validation-calibrated-{variant}.jsonl"
    )
    baseline_artifact = json.loads(
        Path("research/experiments/campaign-04/calibration-logistic-threads2.json").read_text()
    )
    baseline = [
        apply(r, baseline_artifact)
        for r in load("/workspace/houndoracle-data/scores/validation-logistic.jsonl")
    ]

    def at85(rows):
        return summary(
            [
                r
                | {
                    "decision": r["decision"]
                    if r["decision"] in ["YES", "NO"] and (r.get("confidence") or 0) >= 0.85
                    else "MAYBE"
                }
                for r in rows
            ]
        )

    b, c = at85(baseline), at85(calibrated)

    def correct_fraction(rows):
        return sum(
            r["decision"] == r["expected"]
            and r["decision"] in ["YES", "NO"]
            and (r.get("confidence") or 0) >= 0.85
            for r in rows
        ) / len(rows)

    gain = correct_fraction(calibrated) - correct_fraction(baseline)
    passed = (
        gain >= 0.02
        and c["accuracy_all_decisions"] >= b["accuracy_all_decisions"]
        and c["selective_risk"] is not None
        and c["selective_risk"] <= b["selective_risk"]
    )
    result = {
        "variant": variant,
        "baseline": b,
        "candidate": c,
        "correct_definitive_gain": gain,
        "screen_passed": passed,
        "decision": "reserve-independent-evaluation" if passed else "reject-screen",
        "seconds": time.perf_counter() - start,
        "fresh_or_locked_scored": False,
        "seed": 1916,
        "C": 1,
        "warning": "Old validation screen only. No independent generalization or live Gemini advantage established.",
    }
    (public / f"{variant}-decision.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
