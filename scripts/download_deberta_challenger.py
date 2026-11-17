"""Download one reviewed pinned ONNX challenger; no executable remote code or public redistribution."""

import hashlib
import json
import urllib.request
from pathlib import Path

MODEL = "cross-encoder/nli-deberta-v3-small"
REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"
EXPECTED = "03c2221313dc0c3eac9cec1f746d1319d33f2c2901fcce1c0f08f4daac9b6dae"


def main():
    directory = Path("/workspace/houndoracle-models/deberta-small")
    directory.mkdir(exist_ok=True)
    manifest = {
        "model_id": MODEL,
        "revision": REVISION,
        "declared_license": "Apache-2.0",
        "pretraining_datasets": ["SNLI", "MultiNLI"],
        "files": {},
        "rights_note": "Model card reviewed. Base model and training-data rights separate; weights excluded from Apache source distribution.",
    }
    total = 0
    for name in ["README.md", "config.json", "tokenizer.json", "onnx/model_quint8_avx2.onnx"]:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        hasher = hashlib.sha256()
        size = 0
        with (
            urllib.request.urlopen(
                f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}", timeout=60
            ) as r,
            path.open("wb") as f,
        ):
            while chunk := r.read(1024 * 1024):
                size += len(chunk)
                total += len(chunk)
                if total > 190000000:
                    raise ValueError("Download size budget exceeded")
                hasher.update(chunk)
                f.write(chunk)
        sha = hasher.hexdigest()
        if name.endswith(".onnx") and sha != EXPECTED:
            raise ValueError("Pinned upstream model integrity mismatch")
        manifest["files"][name] = {"sha256": sha, "bytes": size}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    Path("research/model-comparisons/deberta-provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"downloaded_bytes": total, "model": MODEL, "revision": REVISION}))


if __name__ == "__main__":
    main()
