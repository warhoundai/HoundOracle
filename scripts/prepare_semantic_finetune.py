"""Prepare pinned safe weights and old-development-only tokenized supervision."""

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer


def main():
    base = Path("/workspace/houndoracle-models/minilm-training-base")
    base.mkdir(exist_ok=True)
    original = Path("/workspace/houndoracle-models/minilm")
    for name in ["config.json", "tokenizer.json", "README.md"]:
        shutil.copyfile(original / name, base / name)
    target = base / "model.safetensors"
    expected = "ec9c5b6a08b60d7a01b465c7ed44b496244dfd1bed3274c95875a74cea89ab35"
    if not target.exists():
        url = "https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768/resolve/b95119ce93d3e065de6214e38cd4a97b0f2f2c6d/model.safetensors"
        size = 0
        with urllib.request.urlopen(url, timeout=60) as r, target.open("wb") as f:
            while chunk := r.read(1024 * 1024):
                size += len(chunk)
                if size > 340000000:
                    raise ValueError("Weight download cap exceeded")
                f.write(chunk)
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        raise ValueError("Pinned base safe weight integrity mismatch")
    source = Path("/workspace/houndoracle-data/campaign-02/development.jsonl")
    tokenizer = Tokenizer.from_file(str(original / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=256)
    rows = []
    for line in source.read_text().splitlines():
        r = json.loads(line)
        enc = tokenizer.encode("\n".join(e["text"] for e in r["evidence"]), r["question"])
        rows.append(
            {
                "id": r["id"],
                "ids": enc.ids,
                "mask": enc.attention_mask,
                "types": enc.type_ids,
                "label": {"NO": 0, "YES": 1, "MAYBE": 2}[r["expected"]],
            }
        )
    out = Path("/workspace/houndoracle-data/semantic-finetune-inputs.jsonl")
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    record = {
        "base_id": "cross-encoder/nli-MiniLM2-L6-H768",
        "base_revision": "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d",
        "base_safetensors_sha256": expected,
        "training_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "tokenized_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "training_rows": len(rows),
        "tokenizer_version": __import__("importlib.metadata").metadata.version("tokenizers"),
        "rights": "Base model card Apache-2.0; dataset/base-training rights separate; local research only, no public weights.",
    }
    Path("research/experiments/campaign-06/semantic-training-inputs.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
