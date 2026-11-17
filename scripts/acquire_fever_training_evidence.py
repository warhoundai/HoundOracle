"""Expand independent research data; no training/test claims based on mirror labels alone."""

import hashlib
import json
import urllib.request
from pathlib import Path

from houndoracle.experiments.dataset import MIRROR_REVISION


def main():
    target = Path("/workspace/houndoracle-data/fever-train-sources")
    target.mkdir(exist_ok=True)
    sources = {
        "original.jsonl": "https://fever.ai/download/fever/train.jsonl",
        "gold.jsonl": f"https://huggingface.co/datasets/copenlu/fever_gold_evidence/resolve/{MIRROR_REVISION}/train.jsonl",
        "license.html": "https://fever.ai/download/fever/license.html",
    }
    manifest = {}
    total = 0
    for name, url in sources.items():
        path = target / name
        if not path.exists():
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                path.with_suffix(".partial").open("wb") as f,
            ):
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 170000000:
                        raise ValueError("Download envelope exceeded")
                    f.write(chunk)
            path.with_suffix(".partial").replace(path)
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if (
            name == "gold.jsonl"
            and sha != "bfa7b19109af675e5ce914dd96779b6382ca5c8c60d32c38e26f0072dd32f1c1"
        ):
            raise ValueError("Pinned mirror file integrity mismatch")
        manifest[name] = {"url": url, "bytes": len(raw), "sha256": sha}
        print(json.dumps({"downloaded": name, "bytes": len(raw)}), flush=True)
    manifest["rights"] = {
        "original": "FEVER/Wikipedia attribution-share-alike",
        "mirror": ["CC-BY-SA-3.0", "GPL-3.0"],
        "handling": "Local research only; no source text redistribution in Apache repository; professional redistribution review remains required.",
    }
    (target / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    Path("research/experiments/campaign-06/data-provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
