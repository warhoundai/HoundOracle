"""One fixed CPU-only domain fine-tune. Run in the isolated training environment."""

import hashlib
import importlib.metadata
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from transformers import RobertaForSequenceClassification


def main():
    output = Path("/workspace/houndoracle-models/minilm-vitc-fever-1921")
    if output.exists():
        raise FileExistsError("Preserve fitted checkpoint; never overwrite an experiment")
    output.mkdir()
    seed = 1921
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    base = Path("/workspace/houndoracle-models/minilm-vitc-fever-1920")
    model = RobertaForSequenceClassification.from_pretrained(
        base, local_files_only=True, use_safetensors=True, attn_implementation="eager"
    )
    model.config.use_cache = False
    initial = model.classifier.out_proj.weight.detach().clone()
    rows = [
        json.loads(s)
        for s in Path("/workspace/houndoracle-data/campaign-08-round2/training-token-inputs.jsonl")
        .read_text()
        .splitlines()
    ]
    random.shuffle(rows)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    losses = []
    start = time.perf_counter()
    complete = True
    model.train()
    for offset in range(0, len(rows), 8):
        if time.perf_counter() - start >= 1500:
            complete = False
            break
        batch = rows[offset : offset + 8]
        length = max(len(r["ids"]) for r in batch)
        ids = torch.tensor(
            [r["ids"] + [1] * (length - len(r["ids"])) for r in batch], dtype=torch.long
        )
        mask = torch.tensor(
            [r["mask"] + [0] * (length - len(r["mask"])) for r in batch], dtype=torch.long
        )
        types = torch.tensor(
            [r["types"] + [0] * (length - len(r["types"])) for r in batch], dtype=torch.long
        )
        labels = torch.tensor([r["label"] for r in batch], dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        result = model(input_ids=ids, attention_mask=mask, token_type_ids=types, labels=labels)
        result.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(result.loss.detach()))
        if len(losses) % 25 == 0:
            print(
                json.dumps(
                    {
                        "steps": len(losses),
                        "mean_loss_last25": sum(losses[-25:]) / 25,
                        "seconds": time.perf_counter() - start,
                    }
                ),
                flush=True,
            )
    model.eval()
    model.save_pretrained(output, safe_serialization=True)
    shutil.copyfile(base / "tokenizer.json", output / "tokenizer.json")
    record = {
        "seed": seed,
        "epochs": 1 if complete else None,
        "completed": complete,
        "optimizer": "AdamW",
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "batch_size": 8,
        "max_length": 256,
        "cpu_threads": 2,
        "steps": len(losses),
        "rows": len(rows),
        "training_seconds": time.perf_counter() - start,
        "losses": losses,
        "classifier_update_l2": float(
            torch.linalg.vector_norm(model.classifier.out_proj.weight.detach() - initial)
        ),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["torch", "transformers", "numpy", "safetensors"]
        },
        "checkpoint_sha256": hashlib.sha256(
            (output / "model.safetensors").read_bytes()
        ).hexdigest(),
        "new_cloud_cost_usd": 0,
        "production_qualified": False,
        "note": "Continued from checkpoint1920 with new real VitaminC training cases and replay; AdamW state reset explicitly. No calibration, validation or test labels used for gradient updates.",
    }
    Path(
        "/workspace/HoundOracle/research/experiments/campaign-08/round-2/semantic-training.json"
    ).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "losses"}, indent=2), flush=True)
    if not complete:
        return

    class Export(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.inner(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            ).logits

    destination = output / "onnx/model.onnx"
    destination.parent.mkdir()
    sample = rows[0]
    ids = torch.tensor([sample["ids"]], dtype=torch.long)
    mask = torch.tensor([sample["mask"]], dtype=torch.long)
    types = torch.tensor([sample["types"]], dtype=torch.long)
    wrapper = Export(model).eval()
    torch.onnx.export(
        wrapper,
        (ids, mask, types),
        str(destination),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        external_data=False,
        dynamo=False,
    )
    print(
        json.dumps({"exported": str(destination), "bytes": destination.stat().st_size}), flush=True
    )


if __name__ == "__main__":
    main()
