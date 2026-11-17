"""Verify numerical policy identity on all existing validation/calibration rows before a runtime port."""

import json
from pathlib import Path

import numpy as np

from houndoracle.experiments.head import HeadNLI
from houndoracle.experiments.nli import OnnxNLI


def load(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    head = json.loads(Path("research/experiments/results/trained-aggregators.json").read_text())[
        "logistic"
    ]
    model = HeadNLI(
        OnnxNLI("/workspace/houndoracle-models/minilm", variant="fp32", threads=2), head, "logistic"
    )
    checked = 0
    changed = 0
    max_delta = 0.0
    for split in ["validation", "calibration"]:
        previous = {
            r["id"]: r
            for r in load(Path(f"/workspace/houndoracle-data/scores/{split}-logistic.jsonl"))
        }
        for row in load(Path(f"/workspace/houndoracle-data/campaign-02/{split}.jsonl")):
            result = model.predict(row["question"], row["evidence"])
            old = previous[row["id"]]
            equal = (
                result["candidate"] == old["candidate"]
                and result["raw_score"] == old["raw_score"]
                and result["logits"] == old["logits"]
                and result["truncated"] == old["truncated"]
            )
            changed += not equal
            checked += 1
            if result["logits"] is not None:
                max_delta = max(
                    max_delta, float(np.max(np.abs(np.array(result["logits"]) - old["logits"])))
                )
        print(json.dumps({"split": split, "checked": checked, "changed": changed}), flush=True)
    report = {
        "checked_rows": checked,
        "changed_policy_tuples": changed,
        "max_logit_delta": max_delta,
        "note": "Runtime equivalence check only; no fresh held-out/general production qualification. Never opens locked or campaign05 data.",
    }
    if changed == 0:
        artifact = json.loads(
            Path("research/experiments/results/calibration-logistic.json").read_text()
        )
        artifact["model_config"] = model.config
        artifact["runtime_port_note"] = (
            f"All {checked} existing calibration/validation policy tuples bitwise identical to original scores. Mapping and risk certificate unchanged; production_qualified remains false."
        )
        Path("research/experiments/campaign-04/calibration-logistic-threads2.json").write_text(
            json.dumps(artifact, indent=2) + "\n"
        )
        report["decision"] = "retain-scoped-runtime-port"
    else:
        report["decision"] = "reject-runtime-port-needs-separate-calibration"
    Path("research/experiments/campaign-04/runtime-identity.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
