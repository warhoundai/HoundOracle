"""Independent-label FEVER import with pinned provenance and page-connected splits."""

import hashlib
import json
import urllib.request
from pathlib import Path

MIRROR_REVISION = "a6b8d891d393e97a4efac791afffb2d7de5e57c6"
SOURCES = {
    "original.jsonl": "https://fever.ai/download/fever/shared_task_dev.jsonl",
    "gold.jsonl": f"https://huggingface.co/datasets/copenlu/fever_gold_evidence/resolve/{MIRROR_REVISION}/valid.jsonl",
    "license.html": "https://fever.ai/download/fever/license.html",
}
LABELS = {"SUPPORTS": "YES", "REFUTES": "NO", "NOT ENOUGH INFO": "MAYBE"}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def acquire(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, url in SOURCES.items():
        path = directory / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read(80_000_001)
            if len(data) > 80_000_000:
                raise ValueError("Dataset download exceeds resource bound")
            path.write_bytes(data)
        manifest[name] = {
            "url": url,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    manifest["rights"] = {
        "original": "FEVER/Wikipedia attribution-share-alike; see downloaded license",
        "mirror_declared": ["CC-BY-SA-3.0", "GPL-3.0"],
        "policy": "Local research only. External text and derived datasets are not part of Apache code distribution. No AVeriTeC. Preserve attribution; obtain redistribution review before publication.",
        "attribution": [
            "Thorne et al. NAACL 2018 FEVER",
            "Atanasova, Wright & Augenstein EMNLP 2020",
            "Wikipedia contributors",
        ],
    }
    (directory / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_checked(directory):
    directory = Path(directory)
    original = {
        r["id"]: r for r in map(json.loads, (directory / "original.jsonl").read_text().splitlines())
    }
    accepted, excluded = [], {}
    for row in map(json.loads, (directory / "gold.jsonl").read_text().splitlines()):
        source = original.get(row["original_id"])
        reason = None
        if source is None:
            reason = "not_in_official_partition"
        elif row["claim"] != source["claim"] or row["label"] != source["label"]:
            reason = "label_or_claim_mismatch"
        elif row["label"] not in LABELS:
            reason = "unknown_label"
        pairs = {(e[0], int(e[1])) for e in row["evidence"] if len(e) == 3}
        if reason is None and row["label"] != "NOT ENOUGH INFO":
            sufficient = any(
                all((e[2], e[3]) in pairs for e in group) for group in source["evidence"]
            )
            if not sufficient:
                reason = "incomplete_gold_evidence_set"
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        pages = {e[2] for group in source["evidence"] for e in group if e[2] is not None}
        pages.update(e[0] for e in row["evidence"])
        # Group NEI claims by their leading subject too; every exact claim stays together.
        subject = row["claim"].split(" is ")[0].split(" was ")[0][:100]
        accepted.append(
            {
                "id": "fever:" + str(source["id"]),
                "question": source["claim"],
                "expected": LABELS[source["label"]],
                "original_label": source["label"],
                "source_pages": sorted(pages),
                "subject": subject,
                "evidence": [
                    {
                        "page": e[0],
                        "sentence_id": int(e[1]),
                        "text": e[2],
                        "url": "https://en.wikipedia.org/wiki/" + e[0],
                    }
                    for e in row["evidence"]
                ],
                "category": "multi-hop"
                if min((len(g) for g in source["evidence"]), default=0) > 1
                else "atomic",
                "temporal_context": "Wikipedia June 2017 snapshot; historical evidence-conditioned label, not current-world truth",
                "dataset_license": "CC-BY-SA-3.0; mirror declares GPL-3.0 too",
            }
        )
    unique = {}
    for row in accepted:
        key = row["id"]
        if key in unique:
            pages = sorted(set(unique[key]["source_pages"]) | set(row["source_pages"]))
            choices = [unique[key], row]
            unique[key] = min(choices, key=lambda r: (len(r["evidence"]), digest(r["evidence"])))
            unique[key]["source_pages"] = pages
        else:
            unique[key] = row
    excluded["duplicate_evidence_variants_collapsed"] = len(accepted) - len(unique)
    return list(unique.values()), excluded


def partition(rows, *, seed=1729, targets=None):
    targets = targets or {
        "development": 2000,
        "validation": 480,
        "calibration": 1200,
        "evaluation": 600,
    }
    parent = {}

    def root(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = root(parent[x])
        return parent[x]

    def union(a, b):
        parent[root(a)] = root(b)

    for row in rows:
        key = row["id"]
        for page in row["source_pages"]:
            union(key, "page:" + page)
        union(key, "subject:" + row["subject"])
        union(key, "claim:" + row["question"].casefold())
    groups = {}
    for row in rows:
        groups.setdefault(root(row["id"]), []).append(row)
    # Exclude giant connected components before modeling, not based on outcomes.
    eligible = [g for g in groups.values() if len(g) <= 40]
    eligible.sort(key=lambda g: digest([seed, sorted(r["id"] for r in g)]))
    splits = {name: [] for name in targets}
    used = set()
    # Stratification uses external annotation labels, never model outcomes.
    from collections import Counter

    for name in ["validation", "calibration", "evaluation", "development"]:
        target = targets[name]
        counts = Counter()
        while len(splits[name]) < target:
            options = []
            for index, group in enumerate(eligible):
                if index in used or (name != "development" and len(group) > 20):
                    continue
                if len(splits[name]) + len(group) > target:
                    continue
                labels = Counter(row["expected"] for row in group)
                gain = sum(
                    min(max(0, target / 3 - counts[label]), value)
                    for label, value in labels.items()
                ) / len(group)
                options.append((gain, -index, index, group, labels))
            if not options:
                break
            _, _, index, group, labels = max(options, key=lambda x: (x[0], x[1]))
            group_id = digest(sorted(r["id"] for r in group))
            for row in group:
                row["group_id"] = group_id
            splits[name].extend(group)
            counts.update(labels)
            used.add(index)
    if any(len(splits[name]) < target - 39 for name, target in targets.items()):
        raise ValueError("Insufficient groups: " + str({k: len(v) for k, v in splits.items()}))
    manifest = {
        "seed": seed,
        "split_rule": "Connected components of all gold/original evidence pages, claim and subject; holdout component size <=20, development <=40; unique IDs and class-stratified whole-component packing before model evaluation",
        "source_count": len(rows),
        "excluded_giant_components": sum(len(g) for g in groups.values() if len(g) > 40),
        "splits": {
            name: {
                "count": len(v),
                "groups": len({r["group_id"] for r in v}),
                "ids": [r["id"] for r in v],
                "digest": digest(v),
            }
            for name, v in splits.items()
        },
        "limitations": [
            "Shared unknown events or unrecognized paraphrases may remain; page/subject grouping does not prove independence.",
            "NEI retrieved sentences are auxiliary, not annotated sufficient evidence.",
            "Pretrained models may have seen FEVER; model pretraining leakage cannot be excluded.",
        ],
    }
    return splits, manifest


def build(directory, output):
    if (Path(output) / "manifest.json").exists():
        raise ValueError("Refusing to overwrite frozen partitions; choose a new campaign directory")
    provenance = acquire(directory)
    rows, excluded = load_checked(directory)
    splits, manifest = partition(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest["provenance"] = provenance
    manifest["exclusions"] = excluded
    for name, examples in splits.items():
        (output / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in examples))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
