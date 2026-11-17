"""Seal whole-page-disjoint successor data, excluding every previously scored component."""

import hashlib
import json
from pathlib import Path

from prepare_abstention_data import Components, keys

from houndoracle.experiments.dataset import digest, load_checked


def main():
    target = Path("/workspace/houndoracle-data/campaign-07-fresh")
    if target.exists():
        raise FileExistsError("Fresh panels already sealed")
    old, _ = load_checked("/workspace/houndoracle-data/fever-sources")
    train, _ = load_checked("/workspace/houndoracle-data/fever-train-sources")
    extra, _ = load_checked("/workspace/houndoracle-data/fever-test-sources")
    pool = train + extra
    seen = []
    for path in [
        Path("/workspace/houndoracle-data/campaign-06/risk.jsonl"),
        Path("/workspace/houndoracle-data/campaign-06/locked.jsonl"),
        Path("/workspace/houndoracle-data/campaign-06-balanced/risk-extended.jsonl"),
        Path("/workspace/houndoracle-data/campaign-06-balanced/locked.jsonl"),
        *Path("/workspace/houndoracle-data/campaign-06-successor").glob("*.jsonl"),
    ]:
        seen.extend(json.loads(s) for s in path.read_text().splitlines())
    graph = Components()
    for row in old + pool:
        for key in keys(row):
            graph.union("id:" + row["id"], key)
    blocked = {graph.root("id:" + r["id"]) for r in old + seen}
    groups = {}
    for row in pool:
        group = graph.root("id:" + row["id"])
        if group not in blocked:
            groups.setdefault(group, []).append(row)
    ordered = sorted(
        groups.values(), key=lambda group: digest([1917, sorted(r["id"] for r in group)])
    )
    selected = []
    for group in ordered:
        usable = [r for r in group if sum(len(e["text"]) for e in r["evidence"]) < 5000]
        if usable:
            row = min(usable, key=lambda r: digest([1917, r["id"]]))
            selected.append(row | {"group_id": digest(sorted(r["id"] for r in group))})
        if len(selected) == 500:
            break
    if len(selected) != 500:
        raise ValueError(
            f"Only {len(selected)} unused whole-page components; do not weaken exclusions"
        )
    target.mkdir()
    manifest = {
        "seed": 1917,
        "excluded_prior_unique_ids": len({r["id"] for r in old + seen}),
        "regime": "Whole-page/subject/claim disjoint from ALL prior observed cases; natural residual distribution, not representative web queries.",
        "panels": {},
    }
    for split, rows in [("risk", selected[:300]), ("locked", selected[300:])]:
        path = target / (split + ".jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest["panels"][split] = {
            "n": len(rows),
            "class_counts": {
                label: sum(r["expected"] == label for r in rows) for label in ["YES", "NO", "MAYBE"]
            },
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "ids": [r["id"] for r in rows],
        }
    Path("research/experiments/campaign-07/fresh-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: {a: b for a, b in v.items() if a != "ids"} for k, v in manifest["panels"].items()}
        )
    )


if __name__ == "__main__":
    main()
