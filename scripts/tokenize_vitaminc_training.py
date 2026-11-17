"""Freeze token inputs with the runtime tokenizer, no evaluation labels used."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

from tokenizers import Tokenizer


def main():
    root = Path("/workspace/houndoracle-data/campaign-08")
    target = root / "training-token-inputs.jsonl"
    if target.exists():
        raise FileExistsError("Training inputs already frozen")
    tokenizer = Tokenizer.from_file("/workspace/houndoracle-models/minilm/tokenizer.json")
    tokenizer.enable_truncation(max_length=256)
    source = root / "development.jsonl"
    with target.open("w") as out:
        for row in map(json.loads, source.read_text().splitlines()):
            passage = "\n".join(e["text"] for e in row["evidence"])
            enc = tokenizer.encode(passage, row["question"])
            out.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "ids": enc.ids,
                        "mask": enc.attention_mask,
                        "types": enc.type_ids,
                        "label": {"NO": 0, "YES": 1, "MAYBE": 2}[row["expected"]],
                    }
                )
                + "\n"
            )
    result = {
        "training_rows": 3200,
        "tokenizer_version": importlib.metadata.version("tokenizers"),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "tokens_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "base_safetensors_sha256": hashlib.sha256(
            Path(
                "/workspace/houndoracle-models/minilm-training-base/model.safetensors"
            ).read_bytes()
        ).hexdigest(),
        "training_script_sha256": hashlib.sha256(
            Path("scripts/train_vitaminc_encoder.py").read_bytes()
        ).hexdigest(),
    }
    Path("research/experiments/campaign-08/training-inputs.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
