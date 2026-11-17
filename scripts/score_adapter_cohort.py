"""Guarded local scoring for a selected adapter on separately sealed cohorts."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from train_contrastive_adapters import score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from houndoracle.experiments.low_rank import attach, restore_trainable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["fit", "risk", "locked", "baseline-locked"])
    args = p.parse_args()
    public = Path("research/experiments/campaign-39")
    private = Path("/workspace/houndoracle-data/campaign-39")
    selected = json.loads(Path("research/experiments/campaign-38/screen.json").read_text())
    if not selected["screen_passed"]:
        raise PermissionError("No passing development candidate")
    adapter = Path("/workspace/houndoracle-data/campaign-38/ce-continuation/adapter.safetensors")
    if (
        hashlib.file_digest(adapter.open("rb"), "sha256").hexdigest()
        != selected["training"]["checkpoint_sha256"]
    ):
        raise ValueError("Selected adapter changed")
    manifest = json.loads((public / "dataset-manifest.json").read_text())
    if args.stage != "fit":
        frozen = json.loads((public / "frozen-policy.json").read_text())
        if frozen["adapter_sha256"] != selected["training"]["checkpoint_sha256"]:
            raise ValueError("Frozen candidate changed")
        if (
            args.stage in ["locked", "baseline-locked"]
            and not json.loads((public / "risk-assessment.json").read_text())["qualified80"]
        ):
            raise PermissionError("Fresh risk gate failed; locked cohort must stay sealed")
    if args.stage == "fit":
        source = Path("/workspace/houndoracle-data/campaign-19/calibration-fit.jsonl")
        expected = json.loads(
            Path("research/experiments/campaign-19/dataset-manifest.json").read_text()
        )["panels"]["calibration-fit"]["sha256"]
    else:
        name = "locked" if "locked" in args.stage else "risk"
        source = private / (name + ".jsonl")
        expected = manifest["panels"][name]["sha256"]
    if hashlib.file_digest(source.open("rb"), "sha256").hexdigest() != expected:
        raise ValueError("Sealed input changed")
    out = private / (args.stage + "-scores.jsonl")
    if out.exists():
        raise FileExistsError("Preserve completed cohort observations")
    base = Path("/workspace/houndoracle-models/deberta-vitc-fever-1963")
    if (
        hashlib.file_digest((base / "model.safetensors").open("rb"), "sha256").hexdigest()
        != selected["training"]["base_checkpoint_sha256"]
    ):
        raise ValueError("Base checkpoint changed")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = AutoModelForSequenceClassification.from_pretrained(
        base,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    if args.stage != "baseline-locked":
        attach(model, rank=8, last_layers=4)
        restore_trainable(model, load_file(str(adapter)))
    labels = {int(k): v.lower() for k, v in model.config.id2label.items()}
    rows = [json.loads(s) for s in source.read_text().splitlines()]
    if len({r["group_id"] for r in rows}) != len(rows):
        raise ValueError("Require one query per independent component")
    with out.open("x") as handle:
        for offset in range(0, len(rows), 50):
            for r in score(model, tokenizer, rows[offset : offset + 50], labels):
                handle.write(json.dumps(r) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {"stage": args.stage, "scored": min(offset + 50, len(rows)), "total": len(rows)}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
