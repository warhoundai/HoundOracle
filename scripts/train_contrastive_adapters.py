"""Fixed local paired-adapter comparison; never updates base weights or reads locked data."""

import gc
import hashlib
import importlib.metadata
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from houndoracle.experiments.low_rank import attach, objective, restore_trainable, trainable_state

PUBLIC = Path("research/experiments/campaign-35")
PRIVATE = Path("/workspace/houndoracle-data/campaign-35")
BASE = Path("/workspace/houndoracle-models/deberta-vitc-fever-1963")


def score(model, tokenizer, rows, labels):
    translate = {"contradiction": "NO", "entailment": "YES", "neutral": "MAYBE"}
    results = []
    model.eval()
    with torch.inference_mode():
        for r in rows:
            start = time.perf_counter()
            passage = "\n".join(e["text"] for e in r["evidence"])
            full = tokenizer(passage, r["question"], truncation=False)["input_ids"]
            batch = tokenizer(
                passage, r["question"], truncation=True, max_length=256, return_tensors="pt"
            )
            z = model(**batch).logits[0]
            p = torch.softmax(z, dim=-1)
            choice = int(p.argmax())
            results.append(
                {k: r[k] for k in ["id", "group_id", "expected"]}
                | {
                    "candidate": translate[labels[choice]],
                    "raw_score": float(p[choice]),
                    "logits": z.tolist(),
                    "truncated": len(full) > 256,
                    "token_count": int(batch["input_ids"].shape[1]),
                    "latency_seconds": time.perf_counter() - start,
                    "citation_supported": None,
                }
            )
    return results


def train(
    arm,
    weight,
    inputs,
    protocol,
    *,
    initial_adapter=None,
    parity_references=None,
    maximum_training_seconds=900,
):
    output = PRIVATE / arm
    output.mkdir(exist_ok=False)
    random.seed(1983)
    np.random.seed(1983)
    torch.manual_seed(1983)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=False)
    labels = {int(k): v.lower() for k, v in model.config.id2label.items()}
    target = {v: k for k, v in labels.items()}
    translate = {"NO": "contradiction", "YES": "entailment", "MAYBE": "neutral"}
    targets = attach(model, rank=8, last_layers=4)
    if initial_adapter is not None:
        if (
            hashlib.file_digest(initial_adapter.open("rb"), "sha256").hexdigest()
            != protocol["initial_adapter_sha256"]
        ):
            raise ValueError("Warm-start adapter checksum mismatch")
        restore_trainable(model, load_file(str(initial_adapter)))
    initial = {n: t.clone() for n, t in trainable_state(model).items()}
    parity_rows = inputs["validation"][:3]
    references = (
        parity_references
        if parity_references is not None
        else json.loads(Path("/workspace/houndoracle-data/campaign-29/result.json").read_text())[
            "scores"
        ]["validation"][:3]
    )
    predictions = score(model, tokenizer, parity_rows, labels)
    error = max(
        abs(a - b)
        for r, s in zip(predictions, references, strict=True)
        for a, b in zip(r["logits"], s["logits"], strict=True)
    )
    if error > 0.0001 or [r["candidate"] for r in predictions] != [
        r["candidate"] for r in references
    ]:
        raise ValueError("Initial adapter/base parity failed")
    pairs = list(inputs["pairs"])
    random.Random(1983).shuffle(pairs)
    encoded = []
    for pair in pairs:
        entry = []
        for r in pair:
            item = tokenizer(
                "\n".join(e["text"] for e in r["evidence"]),
                r["question"],
                truncation=True,
                max_length=256,
            )
            entry.append((item, target[translate[r["expected"]]]))
        encoded.append(entry)
    expected_steps = (len(encoded) + 3) // 4
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n
                ],
                "lr": 2e-4,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n
                ],
                "lr": 2e-5,
            },
        ],
        weight_decay=0.01,
    )
    model.train()
    losses = []
    start = time.monotonic()
    completed = True
    for offset in range(0, len(encoded), 4):
        if time.monotonic() - start >= maximum_training_seconds:
            completed = False
            break
        examples = [item for pair in encoded[offset : offset + 4] for item in pair]
        batch = tokenizer.pad([r[0] for r in examples], padding=True, return_tensors="pt")
        gold = torch.tensor([r[1] for r in examples], dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(**batch).logits
        loss, ce, ranking = objective(logits, gold, pair_weight=weight, margin=1)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1)
        optimizer.step()
        losses.append(
            {
                "total": float(loss.detach()),
                "ce": float(ce.detach()),
                "ranking": float(ranking.detach()),
            }
        )
        if len(losses) % 16 == 0:
            event = {
                "arm": arm,
                "steps": len(losses),
                "total_steps": expected_steps,
                "seconds": time.monotonic() - start,
                "last_loss": losses[-1],
            }
            (PRIVATE / "progress.json").write_text(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
    tensors = trainable_state(model)
    save_file(tensors, str(output / "adapter.safetensors"))
    changed = float(sum(torch.sum((t - initial[n]) ** 2) for n, t in tensors.items()).sqrt())
    report = {
        "arm": arm,
        "completed": completed and len(losses) == expected_steps,
        "steps": len(losses),
        "training_seconds": time.monotonic() - start,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "pair_weight": weight,
        "initial_parity_max_error": error,
        "adapter_update_l2": changed,
        "checkpoint_sha256": hashlib.file_digest(
            (output / "adapter.safetensors").open("rb"), "sha256"
        ).hexdigest(),
        "targets": targets,
        "losses": losses,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["torch", "transformers", "safetensors", "numpy"]
        },
        "base_checkpoint_sha256": protocol["base_checkpoint_sha256"],
        "initial_adapter_sha256": protocol.get("initial_adapter_sha256"),
        "production_qualified": False,
        "new_cloud_cost_usd": 0,
    }
    if report["completed"]:
        report["validation_scores"] = score(model, tokenizer, inputs["validation"], labels)
    (output / "result.json").write_text(json.dumps(report) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ["losses", "validation_scores", "targets"]}
        ),
        flush=True,
    )
    del model, optimizer
    gc.collect()
    return report["completed"]


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads((PUBLIC / "protocol.json").read_text())
    if (
        hashlib.file_digest((BASE / "model.safetensors").open("rb"), "sha256").hexdigest()
        != protocol["base_checkpoint_sha256"]
    ):
        raise ValueError("Retained base checkpoint changed")
    if (
        hashlib.file_digest((PRIVATE / "inputs.json").open("rb"), "sha256").hexdigest()
        != protocol["inputs_sha256"]
    ):
        raise ValueError("Frozen supervised inputs changed")
    inputs = json.loads((PRIVATE / "inputs.json").read_text())
    for arm, weight in [("ce-control", 0), ("contrastive", 0.2)]:
        train(arm, weight, inputs, protocol)
    (PRIVATE / "completed.json").write_text(json.dumps({"completed": True}) + "\n")


if __name__ == "__main__":
    main()
