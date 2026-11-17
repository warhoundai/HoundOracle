"""One-question-per-component fresh risk and locked data, disjoint from every old source component."""

import hashlib
import json
from pathlib import Path

from houndoracle.experiments.dataset import digest, load_checked


class Components:
    def __init__(self):
        self.parent = {}
        self.sizes = {}

    def root(self, key):
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a, b):
        a, b = self.root(a), self.root(b)
        if a != b:
            if self.sizes.get(a, 1) < self.sizes.get(b, 1):
                a, b = b, a
            self.parent[b] = a
            self.sizes[a] = self.sizes.get(a, 1) + self.sizes.get(b, 1)


def keys(row):
    return ["page:" + p for p in row["source_pages"]] + [
        "subject:" + row["subject"],
        "claim:" + row["question"].casefold(),
    ]


def main():
    target = Path("/workspace/houndoracle-data/campaign-06")
    if target.exists():
        raise FileExistsError("Do not overwrite frozen splits")
    old, _ = load_checked("/workspace/houndoracle-data/fever-sources")
    rows, excluded = load_checked("/workspace/houndoracle-data/fever-train-sources")
    components = Components()
    for row in old + rows:
        for key in keys(row):
            components.union("id:" + row["id"], key)
    forbidden = {components.root("id:" + r["id"]) for r in old}
    groups = {}
    for row in rows:
        root = components.root("id:" + row["id"])
        if root not in forbidden:
            groups.setdefault(root, []).append(row)
    available = []
    for group in groups.values():
        group.sort(key=lambda r: digest([1910, r["id"]]))
        row = next((r for r in group if sum(len(e["text"]) for e in r["evidence"]) < 5000), None)
        if row is not None:
            row["group_id"] = digest(sorted(r["id"] for r in group))
            available.append(row)
    available.sort(key=lambda r: digest([1910, r["group_id"]]))
    if len(available) < 600:
        raise ValueError(f"Insufficient fresh independent components: {len(available)}")
    target.mkdir()
    manifest = {
        "seed": 1910,
        "source_rows": len(rows),
        "excluded": excluded,
        "eligible_fresh_components": len(available),
        "old_rows_excluded_by_components": len(old),
        "selection": "Hash components, one hash-selected claim/component; no label or prediction filtering; natural residual-component mixture.",
        "splits": {},
    }
    for name, selected in [("risk", available[:400]), ("locked", available[400:600])]:
        path = target / (name + ".jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in selected))
        manifest["splits"][name] = {
            "rows": len(selected),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "ids": [r["id"] for r in selected],
            "class_counts": {
                v: sum(r["expected"] == v for r in selected) for v in ["YES", "NO", "MAYBE"]
            },
        }
    Path("research/experiments/campaign-06/fresh-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "splits"}))
    print(
        json.dumps(
            {k: {a: b for a, b in v.items() if a != "ids"} for k, v in manifest["splits"].items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
