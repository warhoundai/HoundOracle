"""Seal a final holdout excluding every question scored in the invalidated pilot too."""

import json
from collections import Counter
from pathlib import Path

from houndoracle.experiments.dataset import digest, load_checked

root = Path("/workspace/houndoracle-data/campaign-02")
if Path("research/experiments/final-freeze.json").exists():
    raise RuntimeError("Evaluation is frozen; create a new campaign instead of resealing")
rows, _ = load_checked("/workspace/houndoracle-data/fever-sources")
parents = {}


def find(x):
    parents.setdefault(x, x)
    if parents[x] != x:
        parents[x] = find(parents[x])
    return parents[x]


for row in rows:
    for key in ["page:" + p for p in row["source_pages"]] + [
        "subject:" + row["subject"],
        "claim:" + row["question"].casefold(),
    ]:
        parents[find(row["id"])] = find(key)
groups = {}
for row in rows:
    groups.setdefault(find(row["id"]), []).append(row)
forbidden = set()
for split in ["development", "validation", "calibration"]:
    forbidden.update(
        json.loads(line)["id"] for line in (root / (split + ".jsonl")).read_text().splitlines()
    )
for path in Path("/workspace/houndoracle-data/scores-invalidated-v1").glob("*.jsonl"):
    forbidden.update(json.loads(line)["id"] for line in path.read_text().splitlines())
eligible = [g for g in groups.values() if len(g) <= 20 and not any(r["id"] in forbidden for r in g)]
eligible.sort(key=lambda g: digest([1729, sorted(r["id"] for r in g)]))
selected = []
counts = Counter()
used = set()
while len(selected) < 600:
    options = []
    for i, group in enumerate(eligible):
        if i in used or len(selected) + len(group) > 600:
            continue
        labels = Counter(r["expected"] for r in group)
        gain = sum(min(max(0, 200 - counts[label]), n) for label, n in labels.items()) / len(group)
        options.append((gain, -i, i, group, labels))
    if not options:
        break
    _, _, i, group, labels = max(options, key=lambda x: (x[0], x[1]))
    for row in group:
        row["group_id"] = digest(sorted(r["id"] for r in group))
    selected.extend(group)
    counts.update(labels)
    used.add(i)
if len(selected) < 550:
    raise ValueError("Insufficient untouched evaluation data")
(root / "evaluation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected))
manifest = json.loads((root / "manifest.json").read_text())
manifest["evaluation_seal"] = {
    "rule": "Exclude all complete components containing any previously scored original ID, including invalidated v1 runs.",
    "prior_scored_ids_excluded": len(forbidden),
    "models_evaluated_on_this_holdout": False,
}
manifest["splits"]["evaluation"] = {
    "count": len(selected),
    "groups": len({r["group_id"] for r in selected}),
    "ids": [r["id"] for r in selected],
    "digest": digest(selected),
    "labels": dict(counts),
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
Path("research/experiments/dataset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest["evaluation_seal"], indent=2))
print({k: v for k, v in manifest["splits"]["evaluation"].items() if k != "ids"})
