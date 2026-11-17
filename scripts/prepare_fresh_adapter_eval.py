"""Seal fresh source-connected risk/test cohorts before the continuation is assessed."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from prepare_abstention_data import Components
from prepare_scaled_training import norm, text_key

from houndoracle.experiments.dataset import LABELS, digest, load_checked
from houndoracle.experiments.snapshot import wikipedia_revision_reference


def main():
    root = Path("/workspace/houndoracle-data/campaign-39")
    if root.exists():
        raise FileExistsError("Preserve sealed fresh cohorts")
    graph = Components()
    rows = []
    source_hashes = {}
    for part in ["train", "dev", "test"]:
        source = Path("/workspace/houndoracle-data/vitaminc-sources") / (part + ".jsonl")
        source_hashes[part] = hashlib.file_digest(source.open("rb"), "sha256").hexdigest()
        for line in source.open():
            r = json.loads(line)
            if r["revision_type"] != "real" or r.get("FEVER_id"):
                continue
            if not all(
                isinstance(r.get(k), str) and r[k].strip()
                for k in ["unique_id", "case_id", "claim", "evidence", "page"]
            ):
                continue
            anchor = "page:" + norm(r["page"])
            for key in [
                "case:" + r["case_id"],
                "claim:" + text_key(r["claim"]),
                "span:" + text_key(r["evidence"]),
            ]:
                graph.union(anchor, key)
            rows.append(r | {"part": part, "anchor": anchor})
    used = set()
    blocked = set()

    def block(r):
        for p in r.get("source_pages", []):
            blocked.add(graph.root("page:" + norm(p)))
        if r.get("question"):
            blocked.add(graph.root("claim:" + text_key(r["question"])))
        for e in r.get("evidence", []):
            if isinstance(e, dict) and isinstance(e.get("text"), str) and e["text"].strip():
                blocked.add(graph.root("span:" + text_key(e["text"])))

    old, _ = load_checked("/workspace/houndoracle-data/fever-sources")
    for r in old:
        block(r)
    for path in Path("/workspace/houndoracle-data").glob("campaign-*/*.jsonl"):
        if "token" in path.name:
            continue
        for line in path.open():
            r = json.loads(line)
            if isinstance(r, dict) and "group_id" in r:
                used.add(r["group_id"])
                block(r)
    components = defaultdict(list)
    for r in rows:
        components[graph.root(r["anchor"])].append(r)
    pools = {"train": [], "test": []}
    for component, group in components.items():
        parts = {r["part"] for r in group}
        if len(parts) != 1 or component in blocked:
            continue
        part = next(iter(parts))
        if part not in pools:
            continue
        gid = digest(sorted(r["unique_id"] for r in group))
        if gid in used:
            continue
        cases = defaultdict(list)
        for r in group:
            cases[r["case_id"]].append(r)
        eligible = [
            c
            for c in cases.values()
            if len(c) >= 2 and all(len(r["claim"]) + len(r["evidence"]) < 5000 for r in c)
        ]
        if not eligible:
            continue
        case = min(eligible, key=lambda c: digest([1920, c[0]["case_id"]]))
        chosen = min(case, key=lambda r: digest([1987, r["unique_id"]]))
        pools[part].append((gid, chosen))
    for values in pools.values():
        values.sort(key=lambda x: digest([1987, x[0]]))
    needs = {"risk": ("train", 1000), "locked": ("test", 500)}
    if any(len(pools[part]) < n for part, n in needs.values()):
        raise ValueError("Insufficient never-used components; do not shrink evaluation silently")
    root.mkdir()
    public = Path("research/experiments/campaign-39")
    public.mkdir(exist_ok=True)
    manifest = {
        "seed": 1987,
        "source_file_sha256": source_hashes,
        "selection": "Component-uniform hash selection then fixed hash query within case; no label or prediction filtering. Exclude all prior source/claim/span components and previously used component IDs.",
        "snapshot_references": "Explicit oldid URLs; passage-to-upstream match and citation support not independently audited.",
        "license": "VitaminC/Wikipedia attribution/share-alike rights; text private, no AVeriTeC/ANLI or model-generated labels.",
        "protocol": "Only passing38development candidate may enter risk; freeze map on old1000fit; only passing80risk gate may open500locked. No tuning after risk/test. Strict99joint/citation gate separate.",
        "panels": {},
        "production_qualified": False,
    }
    all_groups = []
    for name, (part, n) in needs.items():
        chosen = []
        for gid, r in pools[part][:n]:
            chosen.append(
                {
                    "id": "vitaminc:" + r["unique_id"],
                    "group_id": gid,
                    "question": r["claim"],
                    "expected": LABELS[r["label"]],
                    "source_pages": [r["page"]],
                    "evidence": [
                        {
                            "page": r["page"],
                            "text": r["evidence"],
                            "url": wikipedia_revision_reference(r["page"], r["wiki_revision_id"]),
                            "recorded_revision_id": r["wiki_revision_id"],
                        }
                    ],
                    "benchmark_scope": "Conditional historical snapshot entailment; not current-world truth",
                }
            )
        path = root / (name + ".jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in chosen))
        manifest["panels"][name] = {
            "n": len(chosen),
            "groups": len({r["group_id"] for r in chosen}),
            "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
            "source_partition": part,
        }
        all_groups.append({r["group_id"] for r in chosen})
    if all_groups[0] & all_groups[1] or any(g & used for g in all_groups):
        raise ValueError("Fresh cohort overlap")
    (public / "dataset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "remaining_pool_sizes": {k: len(v) for k, v in pools.items()},
                "panels": manifest["panels"],
                "scored": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
