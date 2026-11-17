"""Produce actual Torch outputs for dynamic ONNX parity, never evaluation labels."""

import json
from pathlib import Path

import torch
from transformers import RobertaForSequenceClassification


def main():
    torch.set_num_threads(2)
    model = RobertaForSequenceClassification.from_pretrained(
        "/workspace/houndoracle-models/minilm-fever-1912",
        local_files_only=True,
        use_safetensors=True,
        attn_implementation="eager",
    ).eval()
    rows = [
        json.loads(s)
        for s in Path("/workspace/houndoracle-data/semantic-finetune-inputs.jsonl")
        .read_text()
        .splitlines()
    ]
    rows.sort(key=lambda r: len(r["ids"]))
    chosen = [rows[i] for i in [0, 10, 100, 500, 1000, 1500, 1900, 1999]]
    batches = [[r] for r in chosen] + [[chosen[0], chosen[-1]], [chosen[2], chosen[4], chosen[6]]]
    tests = []
    for batch in batches:
        length = max(len(r["ids"]) for r in batch)
        values = {
            "input_ids": [r["ids"] + [1] * (length - len(r["ids"])) for r in batch],
            "attention_mask": [r["mask"] + [0] * (length - len(r["mask"])) for r in batch],
            "token_type_ids": [r["types"] + [0] * (length - len(r["types"])) for r in batch],
        }
        with torch.no_grad():
            logits = model(
                **{k: torch.tensor(v, dtype=torch.long) for k, v in values.items()}
            ).logits.tolist()
        tests.append({"inputs": values, "torch_logits": logits})
    Path("/workspace/houndoracle-data/semantic-parity-cases.json").write_text(
        json.dumps(tests) + "\n"
    )
    print(
        json.dumps(
            {
                "parity_cases": len(tests),
                "sequence_lengths": [len(t["inputs"]["input_ids"][0]) for t in tests],
            }
        )
    )


if __name__ == "__main__":
    main()
