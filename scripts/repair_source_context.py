"""Repair a withdrawn input policy without rerunning unaffected deterministic encoder outputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_campaign import analyze
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from houndoracle.experiments.head import head_digest, infer_head
from houndoracle.experiments.nli import OnnxNLI
from houndoracle.experiments.source_context import SourceContextNLI


def load(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=["source-original", "source-finetuned"])
    old = parser.parse_args().variant
    name = old + "-v2"
    output = Path("research/experiments/campaign-07")
    source = Path("/workspace/houndoracle-data/campaign-07") / old
    target = source.with_name(name)
    if target.exists():
        raise FileExistsError("Preserve repaired run")
    target.mkdir()
    fine = old == "source-finetuned"
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
    model = SourceContextNLI(encoder)
    rows = {}
    repairs = []
    hashes = {}
    for split in ["development", "validation", "calibration"]:
        path = source / f"{split}-{old}.jsonl"
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        prior = load(path)
        examples = {
            r["id"]: r for r in load(f"/workspace/houndoracle-data/campaign-02/{split}.jsonl")
        }
        rows[split] = []
        for row in prior:
            example = examples[row["id"]]
            if any(not e["text"].strip() for e in example["evidence"]):
                raw = model.predict(example["question"], example["evidence"])
                repairs.append({"id": row["id"], "split": split})
                result = row | raw | {"model_input_tokens": raw.get("token_count")}
            else:
                logits = row.get("encoder_logits", row["logits"])
                result = row.copy()
                if logits is not None:
                    values = np.array(logits)
                    probs = np.exp(values - values.max())
                    probs /= probs.sum()
                    index = int(probs.argmax())
                    result.update(
                        logits=logits,
                        probabilities=probs.tolist(),
                        raw_score=float(probs[index]),
                        candidate={"entailment": "YES", "contradiction": "NO", "neutral": "MAYBE"}[
                            encoder.labels[index]
                        ],
                    )
            rows[split].append(result)

    def vector(r):
        return (r["logits"] or [0, 0, 1]) + [
            np.log1p(r.get("model_input_tokens") or 0),
            np.log1p(r["evidence_group_count"]),
        ]

    x = np.array([vector(r) for r in rows["development"]])
    y = np.array([{"NO": 0, "YES": 1, "MAYBE": 2}[r["expected"]] for r in rows["development"]])
    scaler = StandardScaler().fit(x)
    head = LogisticRegression(C=1, max_iter=500, random_state=1916).fit(scaler.transform(x), y)
    checkpoint = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "weights": head.coef_.tolist(),
        "biases": head.intercept_.tolist(),
        "classes": head.classes_.tolist(),
    }
    (output / f"{name}-head.json").write_text(json.dumps(checkpoint, indent=2) + "\n")
    config = model.config | {"aggregation_head": name, "head_digest": head_digest(checkpoint)}
    (target / f"model-{name}.json").write_text(json.dumps(config) + "\n")
    for split, records in rows.items():
        (target / f"{split}-{name}.jsonl").write_text(
            "".join(
                json.dumps(
                    infer_head(r, checkpoint, evidence_group_count=r["evidence_group_count"])
                )
                + "\n"
                for r in records
            )
        )
    analyze(target, output, name)
    (output / f"validation-calibrated-{name}.jsonl").replace(
        target / f"validation-calibrated-{name}.jsonl"
    )
    (output / f"{name}-repair.json").write_text(
        json.dumps(
            {
                "withdrawn_variant": old,
                "new_variant": name,
                "source_sha256": hashes,
                "recomputed_rows": repairs,
                "unaffected_encoder_logits_reused": sum(len(r) for r in rows.values())
                - len(repairs),
                "same_head_hyperparameters": True,
                "fresh_data_scored": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"variant": name, "corrected_rows": len(repairs)}), flush=True)


if __name__ == "__main__":
    main()
