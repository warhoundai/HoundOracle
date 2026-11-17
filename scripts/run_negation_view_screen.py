"""One frozen logical-view screen; not a factual-truth probability or public policy."""

import hashlib
import json
import time
from pathlib import Path

import torch
from train_contrastive_adapters import score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from houndoracle.experiments.metrics import paired_comparison, summary
from houndoracle.experiments.negation import combine, negated_view


def main():
    out = Path("research/experiments/campaign-36/screen.json")
    if out.exists():
        raise FileExistsError("Preserve completed screen")
    if not Path("/workspace/houndoracle-data/campaign-35/completed.json").exists():
        raise RuntimeError("Wait for35to finish before starting CPU screen")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    root = Path("/workspace/houndoracle-models/deberta-vitc-fever-1963")
    protocol = json.loads(out.with_name("protocol.json").read_text())
    if (
        hashlib.file_digest((root / "model.safetensors").open("rb"), "sha256").hexdigest()
        != protocol["base_checkpoint_sha256"]
    ):
        raise ValueError("Base checkpoint changed")
    rows = [
        json.loads(s)
        for s in Path("/workspace/houndoracle-data/campaign-09/validation.jsonl")
        .read_text()
        .splitlines()
    ]
    baseline = json.loads(Path("/workspace/houndoracle-data/campaign-29/result.json").read_text())[
        "scores"
    ]["validation"]
    if [(r["id"], r["group_id"], r["expected"]) for r in rows] != [
        (r["id"], r["group_id"], r["expected"]) for r in baseline
    ]:
        raise ValueError("Unpaired baseline")
    model = AutoModelForSequenceClassification.from_pretrained(
        root,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    labels = {int(k): v.lower() for k, v in model.config.id2label.items()}
    if labels != {0: "contradiction", 1: "entailment", 2: "neutral"}:
        raise ValueError("Unexpected class order")
    parity = score(model, tokenizer, rows[:3], labels)
    error = max(
        abs(x - y)
        for a, b in zip(parity, baseline[:3], strict=True)
        for x, y in zip(a["logits"], b["logits"], strict=True)
    )
    if error > 0.0001:
        raise ValueError("Cached original logits mismatch")
    flip = {"YES": "NO", "NO": "YES", "MAYBE": "MAYBE"}
    views = [
        r
        | {
            "question": negated_view(r["question"], declared_proposition=True),
            "expected": flip[r["expected"]],
        }
        for r in rows
    ]
    start = time.monotonic()
    predictions = score(model, tokenizer, views, labels)
    paired = []
    for a, b in zip(baseline, predictions, strict=True):
        r = combine(a["logits"], b["logits"])
        paired.append(
            {k: a[k] for k in ["id", "group_id", "expected"]}
            | r
            | {
                "decision": r["candidate"],
                "latency_seconds": None,
                "truncated": a["truncated"] or b["truncated"],
                "citation_supported": None,
                "negated_candidate": b["candidate"],
            }
        )
    reference = [r | {"decision": r["candidate"], "latency_seconds": None} for r in baseline]
    a, b = summary(reference), summary(paired)
    passed = (
        b["accuracy_all_decisions"] >= a["accuracy_all_decisions"] + 0.02
        and b["definitive_coverage"] >= a["definitive_coverage"] - 0.02
    )
    result = {
        "baseline_raw": a,
        "candidate_raw": b,
        "paired_comparison": paired_comparison(reference, paired, seed=1983, replicates=2000),
        "screen_passed": passed,
        "decision": "retain-for-fresh-qualification"
        if passed
        else "reject-before-new-held-out-data",
        "views_disagree": sum(not r["views_agree"] for r in paired),
        "initial_parity_max_error": error,
        "negated_inference_seconds": time.monotonic() - start,
        "fresh_risk_opened": False,
        "locked_opened": False,
        "production_qualified": False,
        "new_cloud_spend_usd": 0,
        "scope": "207reused claims, not414independent samples. Only negated-view inference timed; original features cached. Scores correlated and uncalibrated. No truth/citation/general-web guarantee.",
        "records": paired,
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
