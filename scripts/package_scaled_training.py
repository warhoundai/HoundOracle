"""Tokenize only declared training data; private archive excludes every heldout record."""

import hashlib
import json
import tarfile
from pathlib import Path

from tokenizers import Tokenizer


def main():
    root = Path("/workspace/houndoracle-data/campaign-19")
    target = root / "training-bundle.tar.gz"
    if target.exists():
        raise FileExistsError("Preserve prepared training inputs")
    base = Path("/workspace/houndoracle-models/minilm-vitc-fever-1921")
    tokenizer = Tokenizer.from_file(str(base / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=256)
    inputs = root / "training-token-inputs.jsonl"
    with inputs.open("w") as out:
        for line in (root / "development.jsonl").open():
            row = json.loads(line)
            enc = tokenizer.encode("\n".join(e["text"] for e in row["evidence"]), row["question"])
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
    manifest = {
        "initial_sha256": {
            name: hashlib.sha256((base / name).read_bytes()).hexdigest()
            for name in ["model.safetensors", "config.json", "tokenizer.json"]
        },
        "token_inputs_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
        "training_source_sha256": hashlib.sha256(
            (root / "development.jsonl").read_bytes()
        ).hexdigest(),
        "rows": sum(1 for _ in inputs.open()),
        "heldout_records_included": False,
    }
    (root / "input-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with tarfile.open(target, "w:gz", compresslevel=1) as archive:
        for name in manifest["initial_sha256"]:
            archive.add(base / name, arcname="initial/" + name)
        archive.add(inputs, arcname="training-token-inputs.jsonl")
        archive.add(root / "input-manifest.json", arcname="input-manifest.json")
        archive.add("scripts/train_scaled_encoder.py", arcname="train.py")
    public = Path("research/experiments/campaign-19")
    (public / "training-inputs.json").write_text(
        json.dumps(
            manifest
            | {
                "archive_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "archive_bytes": target.stat().st_size,
                "training_script_sha256": hashlib.sha256(
                    Path("scripts/train_scaled_encoder.py").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
