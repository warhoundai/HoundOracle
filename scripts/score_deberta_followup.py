"""Offline scoring of one sealed cohort using the retained fixed research checkpoint."""

import argparse
import hashlib
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve completed cohort scoring")
    if (
        hashlib.file_digest(args.input.open("rb"), "sha256").hexdigest()
        != args.expected_input_sha256
    ):
        raise ValueError("Sealed cohort integrity mismatch")
    root = Path("/workspace/houndoracle-models/deberta-vitc-fever-1963")
    if (
        hashlib.file_digest((root / "model.safetensors").open("rb"), "sha256").hexdigest()
        != "c520871a91b6497b628d60aad945b54c34819d8b53c5c3b0bf814fab7d3f3e70"
    ):
        raise ValueError("Checkpoint integrity mismatch")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        root,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    labels = {int(k): v.lower() for k, v in model.config.id2label.items()}
    translate = {"contradiction": "NO", "entailment": "YES", "neutral": "MAYBE"}
    if set(labels.values()) != set(translate):
        raise ValueError("Unexpected label mapping")
    rows = [json.loads(s) for s in args.input.read_text().splitlines()]
    if len({r["group_id"] for r in rows}) != len(rows):
        raise ValueError("Require one query per independent component")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silently replacing a completed observation.
    with args.output.open("x") as out, torch.inference_mode():
        for i, r in enumerate(rows):
            start = time.perf_counter()
            passage = "\n".join(e["text"] for e in r["evidence"])
            full = tokenizer(passage, r["question"], truncation=False)["input_ids"]
            batch = tokenizer(
                passage, r["question"], truncation=True, max_length=256, return_tensors="pt"
            )
            logits = model(**batch).logits[0].cpu()
            probs = torch.softmax(logits, dim=-1)
            pick = int(probs.argmax())
            record = {k: r[k] for k in ["id", "group_id", "expected"]} | {
                "candidate": translate[labels[pick]],
                "raw_score": float(probs[pick]),
                "logits": [
                    float(logits[next(k for k, v in labels.items() if translate[v] == name)])
                    for name in ["NO", "YES", "MAYBE"]
                ],
                "truncated": len(full) > 256,
                "token_count": int(batch["input_ids"].shape[1]),
                "evidence_group_count": len({e["page"] for e in r["evidence"]}),
                "latency_seconds": time.perf_counter() - start,
                "citation_supported": None,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            if (i + 1) % 100 == 0:
                print(json.dumps({"scored": i + 1}), flush=True)
    print(json.dumps({"completed_rows": len(rows), "production_qualified": False}), flush=True)


if __name__ == "__main__":
    main()
