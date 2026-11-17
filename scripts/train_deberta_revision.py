"""One fixed domain-adaptation experiment; no fresh risk or locked data on worker."""

import argparse
import hashlib
import importlib.metadata
import json
import random
import time
from pathlib import Path


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run(root, progress):
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    manifest = json.loads((root / "manifest.json").read_text())
    inputs = json.loads((root / "inputs.json").read_text())
    if digest_file(root / "inputs.json") != manifest["inputs_sha256"]:
        raise ValueError("Input integrity failure")
    random.seed(1963)
    np.random.seed(1963)
    torch.manual_seed(1963)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    mid, rev = manifest["model_id"], manifest["revision"]
    t = time.monotonic()
    file = hf_hub_download(mid, "model.safetensors", revision=rev, token=False)
    if digest_file(file) != manifest["initial_weights_sha256"]:
        raise ValueError("Initial weight integrity failure")
    tokenizer = AutoTokenizer.from_pretrained(
        mid, revision=rev, trust_remote_code=False, token=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        mid,
        revision=rev,
        trust_remote_code=False,
        use_safetensors=True,
        token=False,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    labels = {int(k): v.lower() for k, v in model.config.id2label.items()}
    translate = {"contradiction": "NO", "entailment": "YES", "neutral": "MAYBE"}
    if set(labels.values()) != set(translate):
        raise ValueError("Unexpected label mapping")
    target = {translate[v]: k for k, v in labels.items()}
    model.eval()
    parity = []
    with torch.inference_mode():
        for r in inputs["parity"]:
            batch = tokenizer(
                "\n".join(e["text"] for e in r["evidence"]),
                r["question"],
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            out = model(**batch).logits[0].tolist()
            ref = [r["logits_by_label"][labels[i]] for i in range(3)]
            delta = max(abs(a - b) for a, b in zip(out, ref))
            parity.append(delta)
            if delta > 0.02 or int(np.argmax(out)) != int(np.argmax(ref)):
                raise ValueError("Published ONNX/PyTorch parity failed")
    frozen = 0
    for p in model.get_input_embeddings().parameters():
        p.requires_grad_(False)
        frozen += p.numel()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=2e-5, weight_decay=0.01
    )
    initial = model.classifier.weight.detach().clone()
    rows = inputs["training"]
    encoded = []
    for r in rows:
        item = tokenizer(
            "\n".join(e["text"] for e in r["evidence"]),
            r["question"],
            truncation=True,
            max_length=256,
        )
        item["labels"] = target[r["expected"]]
        encoded.append(item)
    losses = []
    start = time.monotonic()
    completed = True
    model.train()
    for offset in range(0, len(encoded), 8):
        if time.monotonic() - start >= 1800:
            completed = False
            break
        batch = tokenizer.pad(encoded[offset : offset + 8], padding=True, return_tensors="pt")
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        losses.append(float(output.loss.detach()))
        if len(losses) % 25 == 0:
            event = {
                "stage": "training",
                "steps": len(losses),
                "rows": min(offset + 8, len(rows)),
                "mean_loss_last25": sum(losses[-25:]) / 25,
                "training_seconds": time.monotonic() - start,
            }
            print(json.dumps(event), flush=True)
            progress(event)
    model.eval()
    checkpoint = root / "trained"
    checkpoint.mkdir(exist_ok=False)
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    training = {
        "completed": completed,
        "rows": len(rows),
        "steps": len(losses),
        "epochs": 1 if completed else None,
        "seed": 1963,
        "optimizer": "AdamW",
        "learning_rate": 2e-5,
        "batch_size": 8,
        "max_length": 256,
        "frozen_input_embedding_parameters": frozen,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "training_seconds": time.monotonic() - start,
        "acquisition_seconds": start - t,
        "losses": losses,
        "initial_parity_max_error": max(parity),
        "classifier_update_l2": float(
            torch.linalg.vector_norm(model.classifier.weight.detach() - initial)
        ),
        "checkpoint_sha256": digest_file(checkpoint / "model.safetensors"),
        "production_qualified": False,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["torch", "transformers", "numpy", "safetensors"]
        },
    }
    records = {}
    if completed:
        for split in ["validation", "calibration-fit"]:
            scored = []
            with torch.inference_mode():
                for r in inputs[split]:
                    then = time.monotonic()
                    passage = "\n".join(e["text"] for e in r["evidence"])
                    raw = tokenizer(passage, r["question"], truncation=False)["input_ids"]
                    batch = tokenizer(
                        passage, r["question"], truncation=True, max_length=256, return_tensors="pt"
                    )
                    out = model(**batch).logits[0].cpu()
                    probs = torch.softmax(out, dim=-1).tolist()
                    pick = int(np.argmax(probs))
                    scored.append(
                        {k: r[k] for k in ["id", "group_id", "expected"]}
                        | {
                            "candidate": translate[labels[pick]],
                            "raw_score": probs[pick],
                            "logits": [
                                float(
                                    out[next(i for i, v in labels.items() if translate[v] == name)]
                                )
                                for name in ["NO", "YES", "MAYBE"]
                            ],
                            "truncated": len(raw) > 256,
                            "token_count": int(batch["input_ids"].shape[1]),
                            "evidence_group_count": len({e["page"] for e in r["evidence"]}),
                            "latency_seconds": time.monotonic() - then,
                            "citation_supported": None,
                        }
                    )
            records[split] = scored
            progress({"stage": "scored", "split": split, "rows": len(scored)})
    result = {
        "training": training,
        "scores": records,
        "scope": "Private development/calibration-fit only; no fresh risk/test labels or independently qualified decision/citation claims.",
    }
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    import boto3

    client = boto3.client("s3", region_name="us-east-1")
    bucket = "houndoracle-research-751446514671-20261116"

    def progress(event):
        client.put_object(
            Bucket=bucket, Key="nli29/output/progress.json", Body=json.dumps(event).encode()
        )

    checkpoint = run(args.root, progress)
    client.upload_file(str(args.root / "result.json"), bucket, "nli29/output/result.json")
    for file in checkpoint.iterdir():
        if file.is_file():
            client.upload_file(str(file), bucket, "nli29/output/trained/" + file.name)
    progress({"stage": "completed", "checkpoint_uploaded": True})


if __name__ == "__main__":
    main()
