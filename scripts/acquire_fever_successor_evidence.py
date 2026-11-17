"""Pinned additional research cohort; source text stays outside the Apache repository."""

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from houndoracle.experiments.dataset import MIRROR_REVISION, load_checked


def main():
    root = Path("/workspace/houndoracle-data")
    target = root / "fever-test-sources"
    target.mkdir(exist_ok=True)
    url = f"https://huggingface.co/datasets/copenlu/fever_gold_evidence/resolve/{MIRROR_REVISION}/test.jsonl"
    path = target / "gold.jsonl"
    if not path.exists():
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(10_000_001)
        if len(data) > 10_000_000:
            raise ValueError("Research download limit exceeded")
        path.write_bytes(data)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != "3f4c9a9c6919f9a08af2ad2bcb9cc2be77a656169d93ff69a68fbd750c677b9e":
        raise ValueError("Pinned cohort integrity mismatch")
    shutil.copy(root / "fever-sources/original.jsonl", target / "original.jsonl")
    accepted, excluded = load_checked(target)
    provenance = {
        "url": url,
        "sha256": sha,
        "bytes": path.stat().st_size,
        "accepted": len(accepted),
        "excluded": excluded,
        "rights": "Same FEVER/Wikipedia and mirror local-research restrictions; no source text redistributed.",
        "official_labels_sha256": hashlib.sha256(
            (target / "original.jsonl").read_bytes()
        ).hexdigest(),
    }
    Path("research/experiments/campaign-06/successor-source.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance))


if __name__ == "__main__":
    main()
