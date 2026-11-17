"""Supervised controls and independent revision-level calibration; locked data stays sealed."""

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_campaign import bootstrap_risk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from houndoracle.experiments.fit_calibration import certify, diagnostics, fit
from houndoracle.experiments.head import head_digest, infer_head
from houndoracle.experiments.metrics import cluster_interval, summary
from houndoracle.experiments.pipeline import apply


def load(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def quality(rows):
    result = summary(rows)
    result["latency_seconds"] = None
    cases = {}
    for row in rows:
        cases.setdefault(row["case_id"], []).append(row)
    result["complete_case_accuracy"] = sum(
        all(r["decision"] == r["expected"] for r in case) for case in cases.values()
    ) / len(cases)
    result["macro_f1"] = float(
        f1_score(
            [r["expected"] for r in rows],
            [r["decision"] for r in rows],
            labels=["YES", "NO", "MAYBE"],
            average="macro",
            zero_division=0,
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", choices=["original", "contrastive", "fever-finetuned", "continued"]
    )
    parser.add_argument("--round2", action="store_true")
    args = parser.parse_args()
    if args.round2 and args.model not in ["contrastive", "continued"]:
        parser.error("Round2 fits only parent or continued encoders")
    if args.model == "continued" and not args.round2:
        parser.error("continued requires --round2")
    public = Path("research/experiments/campaign-08")
    if args.round2:
        public = public / "round-2"
    root = (
        Path(
            "/workspace/houndoracle-data/campaign-08-round2/scores"
            if args.round2
            else "/workspace/houndoracle-data/campaign-08/scores"
        )
        / args.model
    )
    config = json.loads((root / "model.json").read_text())
    raw = {
        split: load(root / (split + ".jsonl"))
        for split in ["validation", "calibration-fit", "calibration-risk"]
    }
    sealed_path = public / (
        "dataset-manifest-extended.json"
        if (public / "dataset-manifest-extended.json").exists()
        else "dataset-manifest.json"
    )
    sealed = json.loads(sealed_path.read_text())
    for split, records in raw.items():
        expected = set(sealed["panels"][split]["ids"])
        actual = [r["id"] for r in records]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Incomplete or mismatched evaluation trace")
    heads = {}
    if args.model != "fever-finetuned":
        training = load(root / "development.jsonl")
        if {r["id"] for r in training} != set(sealed["panels"]["development"]["ids"]):
            raise ValueError("Incomplete training feature trace")
        vectors = lambda r: (
            (r["logits"] or [0, 0, 1])
            + [np.log1p(r.get("model_input_tokens") or 0), np.log1p(r["evidence_group_count"])]
        )
        x = np.array([vectors(r) for r in training])
        y = np.array([{"NO": 0, "YES": 1, "MAYBE": 2}[r["expected"]] for r in training])
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(C=1, max_iter=500, random_state=1920).fit(scaler.transform(x), y)
        output_name = (
            ("parent" if args.model == "contrastive" else "continued") + "-mixture"
            if args.round2
            else args.model + "-mixture"
        )
        heads[output_name] = {
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "weights": model.coef_.tolist(),
            "biases": model.intercept_.tolist(),
            "classes": model.classes_.tolist(),
        }
        if args.round2 and args.model == "contrastive":
            heads["parent-frozen"] = json.loads(
                Path("research/experiments/campaign-08/contrastive-mixture-head.json").read_text()
            )
        if args.model == "original":
            heads["original-frozen"] = json.loads(
                Path("research/experiments/results/trained-aggregators.json").read_text()
            )["logistic"]
    else:
        heads["fever-finetuned-frozen"] = json.loads(
            Path("research/experiments/campaign-06/finetuned-head.json").read_text()
        )
    for name, head in heads.items():
        target = public / (name + "-analysis.json")
        if target.exists():
            raise FileExistsError("Preserve independent risk assessment")
        private = root / name
        private.mkdir()
        (public / (name + "-head.json")).write_text(json.dumps(head, indent=2) + "\n")
        scored = {
            split: [
                infer_head(r, head, evidence_group_count=r["evidence_group_count"]) for r in rows
            ]
            for split, rows in raw.items()
        }
        methods = {}
        for method in ["temperature", "logistic", "isotonic"]:
            try:
                mapping = fit(
                    method, [r for r in scored["calibration-fit"] if not r.get("truncated")]
                )
                methods[method] = {
                    "mapping": mapping,
                    "validation": diagnostics(
                        [r for r in scored["validation"] if not r.get("truncated")], mapping
                    ),
                }
            except ValueError as exc:
                methods[method] = {"failure": str(exc)}
        choices = {
            k: v
            for k, v in methods.items()
            if "mapping" in v and v["validation"]["brier"] is not None
        }
        if not choices:
            raise ValueError("No valid correctness mapping")
        selected = min(choices, key=lambda m: choices[m]["validation"]["brier"])
        mapping = choices[selected]["mapping"]
        risk = [r for r in scored["calibration-risk"] if not r.get("truncated")]
        certificate = certify(mapping, risk, alpha=0.05 / 4)
        certificate["row_risk_upper_bootstrap"] = bootstrap_risk(risk, mapping, seed=1920)
        certificate["qualified"] = bool(
            certificate["qualified"]
            and certificate["selected_rows"] >= 30
            and certificate["selected_groups"] >= 20
            and certificate["row_risk_upper_bootstrap"] <= 0.2
        )
        certificate["production_qualified"] = False
        artifact = {
            "schema_version": 1,
            "population": "vitaminc-real-human-revision-pairs-1921"
            if args.round2
            else "vitaminc-real-human-revision-pairs-1920",
            "model_config": config | {"aggregation_head": name, "head_digest": head_digest(head)},
            "mapping": mapping,
            "certificate": certificate,
            "production_qualified": False,
            "scope": "Conditional dataset-label correctness on supplied historical revision evidence, not current-world truth or arbitrary retrieval.",
        }
        (public / (name + "-calibration.json")).write_text(json.dumps(artifact, indent=2) + "\n")
        outputs = {split: [apply(r, artifact) for r in rows] for split, rows in scored.items()}
        for split, rows in outputs.items():
            (private / (split + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
        val = outputs["validation"]
        at85 = [
            r | {"decision": r["decision"] if (r.get("confidence") or 0) >= 0.85 else "MAYBE"}
            for r in val
        ]
        report = {
            "variant": name,
            "certificate": certificate,
            "selected_mapping": selected,
            "methods": methods,
            "raw_validation": quality(
                [r | {"decision": r["candidate"]} for r in scored["validation"]]
            ),
            "calibrated_validation_080": quality(val),
            "calibrated_validation_085": quality(at85),
            "validation_grouped_intervals": {
                key: cluster_interval(val, key, seed=1920)
                for key in ["accuracy_all_decisions", "definitive_coverage", "selective_risk"]
            },
            "independent_risk": quality(outputs["calibration-risk"]),
            "risk_calibration": diagnostics(risk, mapping),
            "locked_scored": False,
            "new_cloud_cost_usd": 0,
            "operational_cost": None,
            "latency_comparison": None,
        }
        target.write_text(json.dumps(report, indent=2) + "\n")
        safe = []
        for split, rows in outputs.items():
            safe.extend(
                {
                    k: r.get(k)
                    for k in [
                        "id",
                        "group_id",
                        "case_id",
                        "expected",
                        "candidate",
                        "decision",
                        "confidence",
                        "raw_score",
                        "truncated",
                    ]
                }
                | {"split": split, "variant": name}
                for r in rows
            )
        Path(
            "benchmarks/results/campaign-08/round-2"
            if args.round2
            else "benchmarks/results/campaign-08",
            name + "-records.jsonl",
        ).write_text("".join(json.dumps(r) + "\n" for r in safe))
        print(
            json.dumps(
                {
                    "variant": name,
                    "qualified": certificate["qualified"],
                    "raw_accuracy": report["raw_validation"]["accuracy_all_decisions"],
                    "raw_case_accuracy": report["raw_validation"]["complete_case_accuracy"],
                    "coverage080": report["calibrated_validation_080"]["definitive_coverage"],
                    "correct_definitive080": report["calibrated_validation_080"][
                        "correct_definitive_fraction"
                    ],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
