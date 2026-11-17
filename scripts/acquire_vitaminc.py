"""Acquire pinned public contrastive evidence data; no source text enters the code repository."""

import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "be6febb761b0b2807687e61e0b5282e459df2fa0"
FILES = {
    "train.jsonl": (193958244, "7461c6fd1a13459590317c5ccdc8651dd2daf7c1ad8ae4b10ccd88d164fccd5a"),
    "dev.jsonl": (32876983, "544934677f5d133873e6d38f4557f8966f4efa5d3d70874ffe6913f2091b86b5"),
    "test.jsonl": (28717099, "7ad1808dbc30c62e0a1427a53022d0dfaff668a1fde3c4b612a2d266edd753ad"),
}


def main():
    root = Path("/workspace/houndoracle-data/vitaminc-sources")
    root.mkdir(exist_ok=True)
    manifest = {
        "revision": REVISION,
        "files": {},
        "rights": {
            "data": "CC-BY-SA-3.0 / applicable Wikipedia article terms; synthetic annotations also derive from FEVER.",
            "code": "Upstream code MIT, separate from dataset rights.",
            "handling": "Local research; preserve attribution. No dataset text or neural weights redistributed in Apache code repository.",
            "attribution": [
                "Schuster, Fisch & Barzilay, NAACL2021, Get Your Vitamin C!",
                "Wikipedia contributors",
            ],
            "paper": "https://aclanthology.org/2021.naacl-main.52/",
        },
    }
    for name, (size, sha) in FILES.items():
        url = f"https://huggingface.co/datasets/tals/vitaminc/resolve/{REVISION}/{name}"
        path = root / name
        if not path.exists():
            total = 0
            with (
                urllib.request.urlopen(url, timeout=45) as response,
                path.with_suffix(".partial").open("wb") as out,
            ):
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > size:
                        raise ValueError("Pinned public download exceeds expected size")
                    out.write(block)
            path.with_suffix(".partial").replace(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != sha or path.stat().st_size != size:
            raise ValueError("Pinned public data integrity mismatch")
        manifest["files"][name] = {"url": url, "bytes": size, "sha256": sha}
        print(json.dumps({"file": name, "bytes": size, "verified": True}), flush=True)
    for name in ["LICENSE", "paper.pdf"]:
        path = root / name
        manifest["files"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    Path("research/experiments/campaign-08/data-provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
