"""Staged scaled-training assessment: development, fresh risk, then frozen locked test."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from houndoracle.experiments.fit_calibration import certify, diagnostics, fit
from houndoracle.experiments.head import head_digest, infer_head
from houndoracle.experiments.metrics import paired_comparison, summary
from houndoracle.experiments.nli import OnnxNLI
from houndoracle.experiments.paired_risk import paired_improvement_lower
from houndoracle.experiments.pipeline import apply

ROOT = Path("/workspace/houndoracle-data/campaign-19")
PUBLIC = Path("research/experiments/campaign-19")
SAFE = Path("benchmarks/results/campaign-19")


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


def save(path, value):
    if path.exists():
        raise FileExistsError("Preserve completed experiment artifact: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def model(name):
    if name == "candidate":
        directory = Path("/workspace/houndoracle-models/minilm-vitc-scaled-1935")
        training = json.loads((directory / "training.json").read_text())
        parity = json.loads((directory / "parity.json").read_text())
        if training["completed"] is not True or parity["passed"] is not True:
            raise ValueError("Partial/unverified training cannot qualify")
        manifest = json.loads((directory / "manifest.json").read_text())
    else:
        directory = Path("/workspace/houndoracle-models/minilm-vitc-fever-1921")
        manifest = json.loads(
            Path("research/experiments/campaign-08/round-2/trained-manifest.json").read_text()
        )
    return OnnxNLI(directory, variant="fp32", threads=1, max_length=256, reviewed_manifest=manifest)


def score(name, split, *, head_training=False):
    source = (
        Path("/workspace/houndoracle-data/campaign-08-round2/development.jsonl")
        if head_training
        else ROOT / (split + ".jsonl")
    )
    rows = load(source)
    if not head_training:
        seal = json.loads((PUBLIC / "dataset-manifest.json").read_text())["panels"][split]
        if hashlib.sha256(source.read_bytes()).hexdigest() != seal["sha256"]:
            raise ValueError("Sealed data changed")
    target = ROOT / "scores" / name / (split + ".jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    config_path = target.parent / "model.json"
    predictor = model(name)
    if config_path.exists():
        if json.loads(config_path.read_text()) != predictor.config:
            raise ValueError("Encoder configuration changed")
    else:
        save(config_path, predictor.config)
    previous = load(target) if target.exists() else []
    if len(previous) > len(rows) or any(a["id"] != b["id"] for a, b in zip(previous, rows)):
        raise ValueError("Existing scores are not an exact prefix")
    # Local inference can resume an unchanged prefix; paid requests are not retried here.
    with target.open("a") as out:
        for index, row in enumerate(rows[len(previous) :], len(previous)):
            result = predictor.predict(row["question"], row["evidence"])
            record = (
                {k: row[k] for k in ["id", "group_id", "expected"]}
                | result
                | {
                    "evidence_group_count": len({e["page"] for e in row["evidence"]}),
                    "latency_seconds": None,
                    "all_in_cost_usd": None,
                    "citation_supported": None,
                }
            )
            out.write(json.dumps(record) + "\n")
            if (index + 1) % 400 == 0:
                print(json.dumps({"model": name, "split": split, "n": index + 1}), flush=True)
    return load(target), predictor.config


def headed(rows, checkpoint):
    return [infer_head(r, checkpoint, evidence_group_count=r["evidence_group_count"]) for r in rows]


def raw_quality(rows):
    return summary([r | {"decision": r["candidate"]} for r in rows])


def provisional(rows, mapping):
    # Provisional development selection only, never a public calibrated result.
    from houndoracle.experiments.fit_calibration import predict

    return summary(
        [
            r
            | {
                "decision": r["candidate"]
                if r["candidate"] in ["YES", "NO"]
                and r.get("raw_score") is not None
                and not r.get("truncated")
                and predict(mapping, r["raw_score"]) >= 0.8
                else "MAYBE"
            }
            for r in rows
        ]
    )


def screen():
    if (PUBLIC / "screen.json").exists():
        raise FileExistsError("Development screen is already final")
    train, config = score("candidate", "head-training", head_training=True)
    values = lambda r: (
        (r["logits"] or [0, 0, 1])
        + [np.log1p(r.get("token_count") or 0), np.log1p(r["evidence_group_count"])]
    )
    x = np.array([values(r) for r in train])
    y = [{"NO": 0, "YES": 1, "MAYBE": 2}[r["expected"]] for r in train]
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(C=1, max_iter=500, random_state=1935).fit(
        scaler.transform(x), y
    )
    checkpoint = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "weights": classifier.coef_.tolist(),
        "biases": classifier.intercept_.tolist(),
        "classes": classifier.classes_.tolist(),
    }
    save(PUBLIC / "candidate-head.json", checkpoint)
    val, _ = score("candidate", "validation")
    fitting, _ = score("candidate", "calibration-fit")
    val = headed(val, checkpoint)
    fitting = headed(fitting, checkpoint)
    methods = {}
    for method in ["temperature", "logistic", "isotonic"]:
        mapping = fit(method, [r for r in fitting if not r.get("truncated")])
        methods[method] = {
            "mapping": mapping,
            "validation": diagnostics([r for r in val if not r.get("truncated")], mapping),
        }
    selected = min(methods, key=lambda m: methods[m]["validation"]["brier"])
    mapping = methods[selected]["mapping"]
    original = json.loads(
        Path("research/experiments/campaign-09/continued-mixture-calibration.json").read_text()
    )
    basehead = json.loads(
        Path("research/experiments/campaign-09/continued-mixture-head.json").read_text()
    )
    baseline = headed(
        load("/workspace/houndoracle-data/campaign-09/scores/continued/validation.jsonl"), basehead
    )
    a, b = raw_quality(baseline), raw_quality(val)
    ac, bc = provisional(baseline, original["mapping"]), provisional(val, mapping)
    raw_delta = b["accuracy_all_decisions"] - a["accuracy_all_decisions"]
    correct_delta = bc["correct_definitive_fraction"] - ac["correct_definitive_fraction"]
    passed = bool(
        (raw_delta >= 0.02 or correct_delta >= 0.02)
        and raw_delta >= -0.01
        and bc["definitive_coverage"] >= ac["definitive_coverage"] - 0.02
    )
    artifact = {
        "schema_version": 1,
        "population": "vitaminc-component-uniform-query-1935",
        "model_config": config
        | {"aggregation_head": "scaled-mixture", "head_digest": head_digest(checkpoint)},
        "mapping": mapping,
        "certificate": {"qualified": False},
        "production_qualified": False,
    }
    save(PUBLIC / "candidate-provisional-calibration.json", artifact)
    save(PUBLIC / "baseline-head.json", basehead)
    save(
        PUBLIC / "baseline-provisional-calibration.json",
        original | {"population": artifact["population"], "certificate": {"qualified": False}},
    )
    report = {
        "raw_baseline": a,
        "raw_candidate": b,
        "provisional_baseline": ac,
        "provisional_candidate": bc,
        "raw_accuracy_delta": raw_delta,
        "correct_definitive_delta": correct_delta,
        "selected_mapping": selected,
        "mapping_methods": methods,
        "screen_passed": passed,
        "locked_scored": False,
        "decision": "qualify-fresh-risk" if passed else "reject-before-fresh-risk",
        "scope": "Reused development; no independent promotion or calibrated public decisions yet.",
    }
    save(PUBLIC / "screen.json", report)
    print(json.dumps(report, indent=2), flush=True)


def qualify():
    screen_report = json.loads((PUBLIC / "screen.json").read_text())
    if screen_report["screen_passed"] is not True:
        raise PermissionError("Failed development screen; do not consume fresh risk/test")
    results = {}
    for name in ["baseline", "candidate"]:
        checkpoint = json.loads((PUBLIC / (name + "-head.json")).read_text())
        artifact = json.loads((PUBLIC / (name + "-provisional-calibration.json")).read_text())
        risk, config = score(name, "calibration-risk")
        risk = headed(risk, checkpoint)
        if config != {
            k: v
            for k, v in artifact["model_config"].items()
            if k not in ["aggregation_head", "head_digest"]
        }:
            raise ValueError("Calibrated model mismatch")
        if len({r["group_id"] for r in risk}) != len(risk):
            raise ValueError("Risk rows are not independent components")
        certificate = certify(
            artifact["mapping"], [r for r in risk if not r.get("truncated")], alpha=0.05 / 2
        )
        certificate["qualified"] = bool(
            certificate["qualified"] and certificate["selected_rows"] >= 30
        )
        certificate["row_risk_upper"] = 1 - certificate["group_correctness_lower_bound"]
        certificate["scope"] = (
            "One query per independent sampled component, historical conditional evidence labels; no per-input or distribution-shift guarantee."
        )
        artifact["certificate"] = certificate
        save(PUBLIC / (name + "-calibration.json"), artifact)
        results[name] = certificate
    frozen = [
        "dataset-manifest.json",
        "screen.json",
        "candidate-head.json",
        "baseline-head.json",
        "candidate-calibration.json",
        "baseline-calibration.json",
    ]
    gate = {
        "qualified": results["candidate"]["qualified"] is True
        and results["baseline"]["qualified"] is True,
        "candidate_preselected": "scaled-mixture",
        "certificates": results,
        "file_sha256": {n: hashlib.sha256((PUBLIC / n).read_bytes()).hexdigest() for n in frozen},
        "locked_seen": False,
        "production_qualified": False,
    }
    save(PUBLIC / "locked-authorization.json", gate)
    print(json.dumps(gate, indent=2), flush=True)


def locked():
    gate = json.loads((PUBLIC / "locked-authorization.json").read_text())
    if gate["qualified"] is not True:
        raise PermissionError("No qualified frozen challenger")
    for name, sha in gate["file_sha256"].items():
        if hashlib.sha256((PUBLIC / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Frozen selection changed")
    rows_by = {}
    outcomes = {}
    safe = []
    for name in ["baseline", "candidate"]:
        checkpoint = json.loads((PUBLIC / (name + "-head.json")).read_text())
        artifact = json.loads((PUBLIC / (name + "-calibration.json")).read_text())
        raw, _ = score(name, "locked")
        raw = headed(raw, checkpoint)
        cal = [apply(r, artifact) for r in raw]
        accepted = [r for r in cal if r["decision"] in ["YES", "NO"]]
        errors = sum(r["decision"] != r["expected"] for r in accepted)
        upper = (
            float(beta.ppf(1 - 0.05 / 2, errors + 1, len(accepted) - errors))
            if accepted and errors < len(accepted)
            else 1.0
        )
        outcomes[name] = {
            "raw": raw_quality(raw),
            "calibrated": summary(cal),
            "accepted_errors": errors,
            "selected_error_family_upper": upper,
        }
        rows_by[name] = {"raw": {r["id"]: r for r in raw}, "cal": {r["id"]: r for r in cal}}
        safe.extend(
            {
                k: r.get(k)
                for k in [
                    "id",
                    "group_id",
                    "expected",
                    "candidate",
                    "decision",
                    "raw_score",
                    "confidence",
                    "calibrated_correctness",
                    "truncated",
                ]
            }
            | {"system": name}
            for r in cal
        )
    keys = sorted(rows_by["baseline"]["raw"])
    if set(keys) != set(rows_by["candidate"]["raw"]):
        raise ValueError("Incomplete matched test")
    delta_a = [
        int(
            rows_by["candidate"]["raw"][k]["candidate"]
            == rows_by["candidate"]["raw"][k]["expected"]
        )
        - int(
            rows_by["baseline"]["raw"][k]["candidate"] == rows_by["baseline"]["raw"][k]["expected"]
        )
        for k in keys
    ]
    delta_k = [
        int(rows_by["candidate"]["cal"][k]["decision"] in ["YES", "NO"])
        - int(rows_by["baseline"]["cal"][k]["decision"] in ["YES", "NO"])
        for k in keys
    ]
    lower_a = paired_improvement_lower(delta_a)
    lower_k = paired_improvement_lower(delta_k)
    passed = bool(
        lower_a >= -0.01
        and lower_k >= -0.02
        and outcomes["candidate"]["selected_error_family_upper"] <= 0.20
    )
    report = {
        "systems": outcomes,
        "raw_accuracy_delta": float(np.mean(delta_a)),
        "coverage_delta": float(np.mean(delta_k)),
        "raw_accuracy_one_sided_95_exact_lower": lower_a,
        "coverage_one_sided_95_exact_lower": lower_k,
        "calibrated_paired_descriptive_intervals": paired_comparison(
            list(rows_by["baseline"]["cal"].values()),
            list(rows_by["candidate"]["cal"].values()),
            seed=1935,
        ),
        "conditional_candidate_validated": passed,
        "production_qualified": False,
        "decision": "retain-scoped-research-challenger"
        if passed
        else "reject-or-investigate-further-no-stable-replacement",
        "scope": "Fresh independent conditional snapshot query population; no Gemini, all-in, live retrieval or public release qualification.",
    }
    save(SAFE / "locked-summary.json", report)
    (SAFE / "locked-records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in safe))
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["screen", "qualify", "locked"])
    args = parser.parse_args()
    {"screen": screen, "qualify": qualify, "locked": locked}[args.stage]()


if __name__ == "__main__":
    main()
