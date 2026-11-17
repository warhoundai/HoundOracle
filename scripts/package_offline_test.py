"""Private task-owned source bundle; exclude credentials and downloaded models/datasets."""

import hashlib
import importlib.metadata
import json
import subprocess
import tarfile
from pathlib import Path


def main():
    root = Path("/workspace/HoundOracle")
    work = Path("/workspace/houndoracle-data/local22")
    work.mkdir(exist_ok=True)
    versions = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"].lower() != "houndoracle"
    }
    (work / "constraints.txt").write_text(
        "\n".join(f"{k}=={v}" for k, v in sorted(versions.items())) + "\n"
    )
    questions = [
        json.loads(l)
        for l in Path("/workspace/houndoracle-data/campaign-09/locked.jsonl")
        .read_text()
        .splitlines()
    ]
    frozen = {
        r["id"]: r["decision"]
        for r in map(
            json.loads,
            Path("benchmarks/results/campaign-09/locked-records.jsonl").read_text().splitlines(),
        )
        if r["variant"] == "continued-mixture"
    }
    payload = {
        "questions": [r | {"frozen_decision": frozen[r["id"]]} for r in questions],
        "manifest": json.loads(
            Path("research/experiments/campaign-08/round-2/trained-manifest.json").read_text()
        ),
        "head": json.loads(
            Path("research/experiments/campaign-09/continued-mixture-head.json").read_text()
        ),
        "calibration": json.loads(
            Path("research/experiments/campaign-09/continued-mixture-calibration.json").read_text()
        ),
    }
    (work / "benchmark.json").write_text(json.dumps(payload) + "\n")
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], text=True
    ).split("\0")
    secret = Path("/workspace/houndoracle-secrets/exa.key").read_bytes().strip()
    target = work / "source.tar.gz"
    with tarfile.open(target, "w:gz", compresslevel=1) as archive:
        for name in sorted(set(names)):
            p = root / name
            if not name or not p.is_file():
                continue
            if secret and secret in p.read_bytes():
                raise ValueError("Credential found in source bundle")
            archive.add(p, arcname="source/" + name)
        archive.add(work / "constraints.txt", arcname="constraints.txt")
        archive.add(work / "benchmark.json", arcname="benchmark.json")
    record = {
        "archive_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "archive_bytes": target.stat().st_size,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_changes_included": True,
        "credentials_included": False,
        "scope": "Private offline deployment regression on fixed previously evaluated600 queries, not new gold-data evaluation.",
    }
    Path("research/experiments/local-22/source-bundle.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
