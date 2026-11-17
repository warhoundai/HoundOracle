"""Frozen identity-head ablation, calibrated without new training or test access."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluate_scaled_training import score

from houndoracle.experiments.fit_calibration import certify, diagnostics, fit
from houndoracle.experiments.head import head_digest, infer_head
from houndoracle.experiments.metrics import summary
from houndoracle.experiments.pipeline import apply

PUBLIC = Path("research/experiments/campaign-20")
ROOT = Path("/workspace/houndoracle-data/campaign-19")
SPECS = {
    "stable-logistic": "baseline",
    "parent-identity": "baseline",
    "scaled-identity": "candidate",
}


def load(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def save(path, value):
    if path.exists():
        raise FileExistsError("Preserve completed ablation stage")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def identity_head():
    return {
        "scaler_mean": [0.0] * 5,
        "scaler_scale": [1.0] * 5,
        "weights": [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        "biases": [0.0, 0.0, 0.0],
        "classes": [0, 1, 2],
    }


def headed(rows, head):
    return [infer_head(r, head, evidence_group_count=r["evidence_group_count"]) for r in rows]


def provisional(rows, mapping):
    # Used only for screening. No risk qualification or public confidence yet.
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
                else "MAYBE",
                "latency_seconds": None,
            }
            for r in rows
        ]
    )


def screen():
    if (PUBLIC / "screen.json").exists():
        raise FileExistsError("Ablation screen is frozen")
    outcomes = {}
    for name, encoder in SPECS.items():
        head = (
            json.loads(
                Path("research/experiments/campaign-09/continued-mixture-head.json").read_text()
            )
            if name == "stable-logistic"
            else identity_head()
        )
        source = (
            Path("/workspace/houndoracle-data/campaign-09/scores/continued")
            if encoder == "baseline"
            else ROOT / "scores/candidate"
        )
        val = headed(load(source / "validation.jsonl"), head)
        fitting = headed(load(source / "calibration-fit.jsonl"), head)
        config = json.loads((source / "model.json").read_text())
        methods = {}
        for method in ["temperature", "logistic", "isotonic"]:
            mapping = fit(method, [r for r in fitting if not r.get("truncated")])
            methods[method] = {
                "mapping": mapping,
                "validation": diagnostics([r for r in val if not r.get("truncated")], mapping),
            }
        chosen = min(methods, key=lambda m: methods[m]["validation"]["brier"])
        mapping = methods[chosen]["mapping"]
        raw = summary([r | {"decision": r["candidate"], "latency_seconds": None} for r in val])
        cal = provisional(val, mapping)
        outcomes[name] = {
            "raw": raw,
            "provisional": cal,
            "selected_mapping": chosen,
            "methods": methods,
        }
        artifact = {
            "schema_version": 1,
            "population": "vitaminc-component-uniform-query-1935",
            "model_config": config | {"aggregation_head": name, "head_digest": head_digest(head)},
            "mapping": mapping,
            "certificate": {"qualified": False},
            "production_qualified": False,
        }
        save(PUBLIC / (name + "-head.json"), head)
        save(PUBLIC / (name + "-provisional-calibration.json"), artifact)
    reference = outcomes["stable-logistic"]
    for name, result in outcomes.items():
        raw_delta = (
            result["raw"]["accuracy_all_decisions"] - reference["raw"]["accuracy_all_decisions"]
        )
        correct_delta = (
            result["provisional"]["correct_definitive_fraction"]
            - reference["provisional"]["correct_definitive_fraction"]
        )
        result["raw_accuracy_delta"] = raw_delta
        result["correct_definitive_delta"] = correct_delta
        result["screen_passed"] = bool(
            name != "stable-logistic"
            and (raw_delta >= 0.02 or correct_delta >= 0.02)
            and raw_delta >= -0.01
            and result["provisional"]["definitive_coverage"]
            >= reference["provisional"]["definitive_coverage"] - 0.02
        )
    result = {
        "systems": outcomes,
        "proceed_to_fresh_risk": any(r["screen_passed"] for r in outcomes.values()),
        "locked_seen": False,
        "production_qualified": False,
        "scope": "Development-only fixed identity ablation; no19 reversal or heldout winner selection.",
    }
    save(PUBLIC / "screen.json", result)
    print(
        json.dumps(
            {
                n: {
                    k: r[k]
                    for k in ["raw_accuracy_delta", "correct_definitive_delta", "screen_passed"]
                }
                | {
                    "raw_accuracy": r["raw"]["accuracy_all_decisions"],
                    "coverage": r["provisional"]["definitive_coverage"],
                }
                for n, r in outcomes.items()
            },
            indent=2,
        )
    )


def qualify():
    screen_report = json.loads((PUBLIC / "screen.json").read_text())
    if screen_report["proceed_to_fresh_risk"] is not True:
        raise PermissionError("No materially promising ablation")
    reports = {}
    raw_by = {}
    for encoder in sorted(set(SPECS.values())):
        raw_by[encoder], _ = score(encoder, "calibration-risk")
    for name, encoder in SPECS.items():
        head = json.loads((PUBLIC / (name + "-head.json")).read_text())
        artifact = json.loads((PUBLIC / (name + "-provisional-calibration.json")).read_text())
        raw = headed(raw_by[encoder], head)
        if len({r["group_id"] for r in raw}) != len(raw):
            raise ValueError("Dependent risk rows")
        certificate = certify(
            artifact["mapping"], [r for r in raw if not r.get("truncated")], alpha=0.05 / 3
        )
        certificate["qualified"] = bool(
            certificate["qualified"] and certificate["selected_rows"] >= 30
        )
        certificate["row_risk_upper"] = 1 - certificate["group_correctness_lower_bound"]
        certificate["scope"] = (
            "Component-uniform one conditional query per independent component, not arbitrary shift or a per-query truth guarantee."
        )
        artifact["certificate"] = certificate
        save(PUBLIC / (name + "-calibration.json"), artifact)
        reports[name] = {
            "certificate": certificate,
            "summary": summary([apply(r, artifact) | {"latency_seconds": None} for r in raw]),
        }
    eligible = [
        n
        for n in SPECS
        if n != "stable-logistic"
        and screen_report["systems"][n]["screen_passed"]
        and reports[n]["certificate"]["qualified"]
    ]
    selected = (
        max(
            eligible,
            key=lambda n: (
                screen_report["systems"][n]["provisional"]["correct_definitive_fraction"],
                screen_report["systems"][n]["raw"]["accuracy_all_decisions"],
            ),
        )
        if eligible
        else None
    )
    files = ["protocol.json", "screen.json"] + [
        n + s for n in SPECS for s in ["-head.json", "-calibration.json"]
    ]
    gate = {
        "qualified": selected is not None,
        "model_selected": selected,
        "systems": reports,
        "file_sha256": {n: hashlib.sha256((PUBLIC / n).read_bytes()).hexdigest() for n in files},
        "source_dataset_manifest_sha256": hashlib.sha256(
            Path("research/experiments/campaign-19/dataset-manifest.json").read_bytes()
        ).hexdigest(),
        "locked_seen": False,
        "production_qualified": False,
    }
    save(PUBLIC / "locked-authorization.json", gate)
    print(json.dumps(gate, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["screen", "qualify"])
    args = parser.parse_args()
    {"screen": screen, "qualify": qualify}[args.stage]()


if __name__ == "__main__":
    main()
