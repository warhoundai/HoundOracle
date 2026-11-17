"""Select candidates on validation; never reads the locked evaluation examples."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from houndoracle.experiments.dataset import digest
from houndoracle.experiments.fit_calibration import certify, diagnostics, fit, predict
from houndoracle.experiments.metrics import cluster_interval, summary
from houndoracle.experiments.pipeline import apply


def load(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def bootstrap_risk(rows, mapping, seed=1729):
    from random import Random

    groups = {}
    for row in rows:
        if (
            row["candidate"] in ("YES", "NO")
            and row["raw_score"] is not None
            and not row.get("truncated")
            and predict(mapping, row["raw_score"]) >= 0.8
        ):
            groups.setdefault(row["group_id"], []).append(row)
    if not groups:
        return 1.0
    rng = Random(seed)
    keys = sorted(groups)
    values = []
    for _ in range(2000):
        sample = [r for g in rng.choices(keys, k=len(keys)) for r in groups[g]]
        values.append(sum(r["candidate"] != r["expected"] for r in sample) / len(sample))
    return float(np.quantile(values, 0.95))


def analyze(scores, output, variant):
    scores, output = Path(scores), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    validation = load(scores / f"validation-{variant}.jsonl")
    calibration = load(scores / f"calibration-{variant}.jsonl")
    if len({r["id"] for r in validation}) != len(validation):
        raise ValueError("Duplicate validation IDs")
    groups = sorted({r["group_id"] for r in calibration}, key=lambda g: digest([1729, g]))
    fit_groups = set(groups[::2])
    risk_groups = set(groups[1::2])
    fitting = [r for r in calibration if r["group_id"] in fit_groups]
    risk = [r for r in calibration if r["group_id"] in risk_groups]
    methods = {}
    for method in ["temperature", "logistic", "isotonic"]:
        try:
            mapping = fit(method, fitting)
            methods[method] = {"mapping": mapping, "validation": diagnostics(validation, mapping)}
        except ValueError as exc:
            methods[method] = {"failure": str(exc)}
    available = {name: data for name, data in methods.items() if "mapping" in data}
    name = min(available, key=lambda key: available[key]["validation"]["brier"])
    mapping = available[name]["mapping"]
    certificate = certify(mapping, risk)
    certificate["row_risk_upper_bootstrap"] = bootstrap_risk(risk, mapping)
    certificate["qualified"] = (
        certificate["selected_groups"] >= 20
        and certificate["selected_rows"] >= 30
        and certificate["row_risk_upper_bootstrap"] <= 0.2
    )
    certificate["qualification_scope"] = (
        "Empirical clustered-row risk gate for this historical population; not a finite-sample guarantee or general production authorization."
    )
    model_config = json.loads((scores / f"model-{variant}.json").read_text())
    artifact = {
        "schema_version": 1,
        "population": "fever-human-labels-deduplicated-v2-small-components-historical",
        "model_config": model_config,
        "mapping": mapping,
        "certificate": certificate,
        "dataset_digest": digest(calibration),
        "fit_group_ids": sorted(fit_groups),
        "risk_group_ids": sorted(risk_groups),
        "validation_digest": digest(validation),
        "production_qualified": False,
        "note": "Final policy uses original supplied evidence, NLI candidate, no truncation, mapping >=.8 and independent risk gate. Production source/semantic policy has not been replaced.",
    }
    results = [apply(r, artifact) for r in validation]
    raw = [{**r, "decision": r["candidate"]} for r in validation]
    report = {
        "variant": variant,
        "raw_summary": summary(raw),
        "calibrated_summary": summary(results),
        "raw_correctness_calibration": diagnostics(validation),
        "methods": methods,
        "selected_mapping": name,
        "risk_calibration_diagnostics": diagnostics(risk, mapping),
        "certificate": certificate,
        "validation_intervals": {
            metric: cluster_interval(results, metric)
            for metric in ["accuracy_all_decisions", "definitive_coverage", "selective_risk"]
        },
    }
    (output / f"calibration-{variant}.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (output / f"analysis-{variant}.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / f"validation-calibrated-{variant}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in results)
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                in [
                    "variant",
                    "raw_summary",
                    "calibrated_summary",
                    "selected_mapping",
                    "certificate",
                ]
            },
            indent=2,
        ),
        flush=True,
    )
    return artifact, report


def train_aggregators(scores, output):
    scores, output = Path(scores), Path(output)
    train = load(scores / "development-fp32.jsonl")
    valid = load(scores / "validation-fp32.jsonl")
    label = {"NO": 0, "YES": 1, "MAYBE": 2}

    def vector(row):
        logits = row["logits"] or [0, 0, 1]
        return logits + [
            np.log1p(row.get("model_input_tokens") or 0),
            np.log1p(row["evidence_group_count"]),
        ]

    x = np.array([vector(r) for r in train])
    y = np.array([label[r["expected"]] for r in train])
    vx = np.array([vector(r) for r in valid])
    vy = np.array([label[r["expected"]] for r in valid])
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    vx = scaler.transform(vx)
    reports = {}
    for name, model in [
        ("logistic", LogisticRegression(C=1, max_iter=500, random_state=1729)),
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(16,),
                learning_rate_init=0.001,
                max_iter=150,
                random_state=1729,
                early_stopping=False,
                alpha=0.01,
            ),
        ),
    ]:
        started = time.perf_counter()
        model.fit(x, y)
        seconds = time.perf_counter() - started
        started = time.perf_counter()
        prob = model.predict_proba(vx)
        prediction = prob.argmax(axis=1)
        latency = (time.perf_counter() - started) / len(vx)
        reports[name] = {
            "training_rows": len(train),
            "training_groups": len({r["group_id"] for r in train}),
            "training_digest": digest(train),
            "seed": 1729,
            "training_seconds": seconds,
            "validation_accuracy": float((prediction == vy).mean()),
            "validation_log_loss": float(log_loss(vy, prob)),
            "training_log_loss": float(log_loss(y, model.predict_proba(x))),
            "aggregator_batch_latency_seconds_per_example": latency,
            "parameters": model.get_params(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "classes": model.classes_.tolist(),
        }
        if name == "mlp":
            reports[name].update(
                weights=[w.tolist() for w in model.coefs_],
                biases=[b.tolist() for b in model.intercepts_],
                epochs=model.n_iter_,
                loss_curve=model.loss_curve_,
                training_loss=float(model.loss_),
            )
        else:
            reports[name].update(weights=model.coef_.tolist(), biases=model.intercept_.tolist())
    (output / "trained-aggregators.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(
        "TRAINED AGGREGATORS",
        json.dumps(
            {
                k: {
                    a: b
                    for a, b in v.items()
                    if a in ["validation_accuracy", "validation_log_loss", "training_seconds"]
                }
                for k, v in reports.items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--variant",
        choices=["fp32", "int8", "logistic", "mlp", "logistic-int8", "mlp-int8"],
        required=True,
    )
    p.add_argument("--train-aggregators", action="store_true")
    args = p.parse_args()
    analyze(args.scores, args.output, args.variant)
    if args.train_aggregators:
        train_aggregators(args.scores, args.output)
