"""Freeze a paired-supervision pilot without opening new held-out data."""

import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    private = Path("/workspace/houndoracle-data/campaign-35")
    if (private / "inputs.json").exists():
        raise FileExistsError("Preserve frozen training inputs")
    rows = [
        json.loads(s)
        for s in Path("/workspace/houndoracle-data/campaign-19/development.jsonl")
        .read_text()
        .splitlines()
    ]
    groups = defaultdict(list)
    for row in rows:
        if row["id"].startswith("vitaminc:") and row.get("case_id"):
            groups[row["case_id"]].append(row)
    pairs = []
    excluded_conflicting_identity = 0
    for items in groups.values():
        choices = []
        for a, b in itertools.combinations(items, 2):
            if a["expected"] == b["expected"]:
                continue
            same_input = (a["question"], a["evidence"]) == (b["question"], b["evidence"])
            if same_input:
                excluded_conflicting_identity += 1
                continue
            choices.append((a, b))
        if choices:
            pairs.append(min(choices, key=lambda p: digest([1983, p[0]["id"], p[1]["id"]])))
    pairs.sort(key=lambda p: digest([1983, p[0]["case_id"]]))
    pairs = pairs[:512]
    if len(pairs) != 512:
        raise ValueError("Insufficient independently labeled contrasting pairs")
    val = [
        json.loads(s)
        for s in Path("/workspace/houndoracle-data/campaign-09/validation.jsonl")
        .read_text()
        .splitlines()
    ]
    cal = [
        json.loads(s)
        for s in Path("/workspace/houndoracle-data/campaign-09/calibration-fit.jsonl")
        .read_text()
        .splitlines()
    ]
    train_groups = {r["group_id"] for p in pairs for r in p}
    if train_groups & {r["group_id"] for r in val + cal}:
        raise ValueError("Train/evaluation source-group overlap")
    inputs = {"pairs": pairs, "validation": val}
    (private / "inputs.json").write_text(json.dumps(inputs) + "\n")
    public = {
        "hypothesis": "Contrastive supervision may improve sensitivity to small factual revisions. Compare identical low-rank adapter training with CE-only versus CE plus pairwise ranking; do not attribute an adapter effect to ranking alone.",
        "base_checkpoint_sha256": "c520871a91b6497b628d60aad945b54c34819d8b53c5c3b0bf814fab7d3f3e70",
        "seed": 1983,
        "training_pairs": 512,
        "training_rows": 1024,
        "source_groups": len(train_groups),
        "training_data": "Human VitaminC real-revision pairs from existing training partition only; no generated labels or new held-out inputs. Some rows were already seen by the base model.",
        "controls": ["CE-only adapter", "CE plus0.2pairwise softplus ranking adapter"],
        "optimizer": "AdamW2e-5 weight_decay.01 gradientclip1",
        "batch_rows": 8,
        "epochs": 1,
        "rank": 8,
        "last_attention_layers": 4,
        "trainable": "Only query/value low-rank factors and classifier; base weights frozen",
        "max_length": 256,
        "pair_margin": 1.0,
        "selection": "Same512pairs and batch order for both arms. Fixed final checkpoint, no intermediate validation selection.",
        "screen": "Reused207development cases; no independent promotion.",
        "gate": "Raw accuracy improvement>=.02 versus frozen29 baseline and raw definitive coverage regression<=.02. Reject incomplete training. If both pass prefer higher correct-definitive fraction, then accuracy, then CE-only.",
        "resources": {
            "local_cpu_threads": 2,
            "local_memory_limit_bytes": 8589934592,
            "maximum_training_seconds_per_arm": 900,
            "new_cloud_spend_usd": 0,
            "large_base_weights_copied": False,
        },
        "future_gate": "Only a passing development candidate may proceed to separately frozen calibration and fresh independent cohorts. Existing consumed600must not be used to tune.",
        "excluded_contradictory_identity_pairs": excluded_conflicting_identity,
        "inputs_sha256": hashlib.file_digest(
            (private / "inputs.json").open("rb"), "sha256"
        ).hexdigest(),
        "production_qualified": False,
    }
    Path("research/experiments/campaign-35/protocol.json").write_text(
        json.dumps(public, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: public[k]
                for k in [
                    "training_pairs",
                    "source_groups",
                    "excluded_contradictory_identity_pairs",
                    "inputs_sha256",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
