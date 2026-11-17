"""Fixed full-class adjustment with disjoint correctness-calibration fitting."""

import hashlib
import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from houndoracle.experiments.class_calibration import SCHEMA, apply
from houndoracle.experiments.fit_calibration import fit, predict
from houndoracle.experiments.metrics import paired_comparison, summary


def raw(rows):
    return [r | {"decision": r["candidate"], "latency_seconds": None} for r in rows]


def selected(rows, mapping):
    return [
        r
        | {
            "decision": r["candidate"]
            if r["candidate"] in ["YES", "NO"]
            and not r["truncated"]
            and predict(mapping, r["raw_score"]) >= 0.8
            else "MAYBE",
            "latency_seconds": None,
        }
        for r in rows
    ]


def main():
    public = Path("research/experiments/campaign-37")
    if (public / "protocol.json").exists():
        raise FileExistsError("Preserve frozen screen; do not rerun tuning")
    protocol = {
        "hypothesis": "Adjusting all three relation logits may correct class bias that a max-score correctness map cannot fix. This is a distinct post-hoc score calibration experiment, not new encoder training.",
        "model": "FrozenDeBERTa29",
        "fit": "Dedicated calibration-fit1000 split deterministically into500class-map fit and500correctness-map fit, source/event-disjoint. Validation207remains separate.",
        "selection": "Single C1standardizedmultinomiallogistic map, seed1985 max_iter1000, no hyperparameter search",
        "features": SCHEMA,
        "gate": "Raw accuracy improvement>=.02 and raw definitive coverage regression<=.02 vs frozen29. No public promotion from development.",
        "confidence": "Fit separate logistic correctness map on second500only. Report provisional80diagnostics, not a guarantee.",
        "resources": "Cached scores, no new model calls or cloud spend",
        "production_qualified": False,
    }
    (public / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    data = json.loads(Path("/workspace/houndoracle-data/campaign-29/result.json").read_text())[
        "scores"
    ]
    cal = sorted(
        data["calibration-fit"],
        key=lambda r: hashlib.sha256(("1985:" + r["group_id"]).encode()).hexdigest(),
    )
    a, b = cal[:500], cal[500:]
    val = data["validation"]
    groups = [{r["group_id"] for r in rows} for rows in [a, b, val]]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Partition leakage")
    scaler = StandardScaler().fit([r["logits"] for r in a])
    labels = {"NO": 0, "YES": 1, "MAYBE": 2}
    model = LogisticRegression(C=1, random_state=1985, max_iter=1000).fit(
        scaler.transform([r["logits"] for r in a]), [labels[r["expected"]] for r in a]
    )
    artifact = {
        "feature_schema": SCHEMA,
        "classes": model.classes_.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "weights": model.coef_.tolist(),
        "bias": model.intercept_.tolist(),
    }
    predictions = [apply(r, artifact) for r in val]
    candidate_map = fit("logistic", [apply(r, artifact) for r in b if not r["truncated"]])
    oldmap = json.loads(
        Path("research/experiments/campaign-29/provisional-mapping.json").read_text()
    )
    old, new = summary(raw(val)), summary(raw(predictions))
    passed = (
        new["accuracy_all_decisions"] >= old["accuracy_all_decisions"] + 0.02
        and new["definitive_coverage"] >= old["definitive_coverage"] - 0.02
    )
    result = {
        "baseline_raw": old,
        "candidate_raw": new,
        "paired_raw": paired_comparison(raw(val), raw(predictions), seed=1985, replicates=2000),
        "baseline_provisional80": summary(selected(val, oldmap)),
        "candidate_provisional80": summary(selected(predictions, candidate_map)),
        "screen_passed": passed,
        "decision": "retain-for-fresh-qualification"
        if passed
        else "reject-before-new-held-out-data",
        "class_fit_count": len(a),
        "correctness_fit_count": len(b),
        "raw_class_counts_fit": {label: sum(r["expected"] == label for r in a) for label in labels},
        "production_qualified": False,
        "fresh_risk_opened": False,
        "locked_opened": False,
        "new_cloud_spend_usd": 0,
        "scope": "Reused development; affine label adjustment and correctness mapping use distinct500-case groups. No audit/current-web/truth-confidence claim.",
        "records": predictions,
    }
    (public / "class-map.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (public / "correctness-map.json").write_text(json.dumps(candidate_map, indent=2) + "\n")
    (public / "screen.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
