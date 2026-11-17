"""Reserve new public-human-annotation components without scoring locked examples.

Candidate pools are NOT audited gold and cannot pass qualification95.seal until
independent source/temporal/licensing audits are completed. Existing39untouched.
"""

import argparse
import hashlib
import json
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from prepare_abstention_data import Components

from houndoracle.experiments.dataset import LABELS, MIRROR_REVISION, digest, load_checked

PUBLIC = Path("research/qualification95")
PRIVATE = Path("/workspace/houndoracle-data/qualification95-candidates")
URLS = {
    "original": "https://fever.ai/download/fever/train.jsonl",
    "mirror": f"https://huggingface.co/datasets/copenlu/fever_gold_evidence/resolve/{MIRROR_REVISION}/train.jsonl",
}


def stream(url, manifest, name):
    size = 0
    h = hashlib.sha256()
    started = time.monotonic()
    with urllib.request.urlopen(url, timeout=45) as response:
        while True:
            line = response.readline(1_000_001)
            if not line:
                break
            size += len(line)
            h.update(line)
            if len(line) > 1_000_000 or size > 400_000_000 or time.monotonic() - started > 240:
                raise ValueError("Source download bound exceeded")
            yield json.loads(line)
    manifest[name] = {"url": url, "bytes": size, "sha256": h.hexdigest()}


def vitamin_pool(blocked_pages, blocked_claims, sources):
    from houndoracle.experiments.snapshot import wikipedia_revision_reference

    normalize = lambda x: " ".join(x.replace("_", " ").casefold().split())
    graph = Components()
    rows = []
    for path in Path("/workspace/houndoracle-data/vitaminc-sources").glob("*.jsonl"):
        sources[path.name] = {
            "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
            "bytes": path.stat().st_size,
        }
        for line in path.open():
            r = json.loads(line)
            if r.get("revision_type") != "real" or r.get("FEVER_id") not in (None, ""):
                continue
            if r.get("label") not in LABELS or not all(
                isinstance(r.get(k), str) and r[k].strip()
                for k in ["page", "claim", "evidence", "case_id", "unique_id"]
            ):
                continue
            try:
                url = wikipedia_revision_reference(r["page"], r["wiki_revision_id"])
            except (ValueError, TypeError, KeyError):
                continue
            anchor = "page:" + normalize(r["page"])
            for key in [
                "case:" + r["case_id"],
                "claim:" + normalize(r["claim"]),
                "span:" + normalize(r["evidence"]),
            ]:
                graph.union(anchor, key)
            rows.append((anchor, r, url))
    blocked = {graph.root("page:" + p) for p in blocked_pages} | {
        graph.root("claim:" + q) for q in blocked_claims
    }
    groups = defaultdict(list)
    for anchor, r, url in rows:
        if graph.root(anchor) not in blocked:
            groups[graph.root(anchor)].append((r, url))
    pool = []
    for component, variants in groups.items():
        r, url = min(variants, key=lambda pair: digest([2003, pair[0]["unique_id"]]))
        pool.append(
            {
                "id": "vitaminc-new95:" + r["unique_id"],
                "group_id": digest([component]),
                "question": r["claim"],
                "expected": LABELS[r["label"]],
                "source_ids": ["wiki:" + normalize(r["page"])],
                "event_id": digest([component]),
                "evidence": [
                    {
                        "url": url,
                        "page": r["page"],
                        "text": r["evidence"],
                        "recorded_revision_id": r["wiki_revision_id"],
                    }
                ],
                "label_origin": "human_dataset",
                "synthetic": False,
                "annotation_reference": "VitaminC:" + r["unique_id"],
                "label_basis": "passage_only_nei"
                if LABELS[r["label"]] == "MAYBE"
                else "original_human_annotation",
                "as_of": "Wikipedia revision " + str(r["wiki_revision_id"]),
                "evidence_scope": "supplied_passages",
                "audit_status": "pending",
                "audit_reference": None,
                "commercial_use_reviewed": False,
                "license_reference": "VitaminC/Wikipedia CC-BY-SA provenance review pending",
                "categories": [],
                "domain": "unclassified",
            }
        )
    return pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["fever", "vitaminc"], default="fever")
    args = parser.parse_args()
    if PRIVATE.exists():
        raise FileExistsError("Preserve reserved candidate pool")
    blocked_pages = set()
    blocked_claims = set()
    normalize = lambda x: " ".join(x.replace("_", " ").casefold().split())
    old, _ = load_checked("/workspace/houndoracle-data/fever-sources")
    for r in old:
        blocked_pages.update(normalize(p) for p in r["source_pages"])
        blocked_claims.add(normalize(r["question"]))
    # Read metadata from actual previous training/evaluation records, not every
    # row in an available-but-unused raw dataset. Sealed39labels are never scored,
    # emitted or used for selection; only its source/component metadata is blocked.
    from urllib.parse import unquote, urlsplit

    metadata_files = 0

    def block(value):
        if isinstance(value, list):
            for item in value:
                block(item)
        elif isinstance(value, dict):
            for page in value.get("source_pages", []) or []:
                if isinstance(page, str):
                    blocked_pages.add(normalize(page))
            if isinstance(value.get("page"), str):
                blocked_pages.add(normalize(value["page"]))
            if isinstance(value.get("url"), str):
                parsed = urlsplit(value["url"])
                if (
                    parsed.hostname
                    and parsed.hostname.endswith("wikipedia.org")
                    and parsed.path.startswith("/wiki/")
                ):
                    blocked_pages.add(normalize(unquote(parsed.path[6:])))
            for field in ["question", "claim"]:
                if isinstance(value.get(field), str):
                    blocked_claims.add(normalize(value[field]))
            for key, item in value.items():
                if key not in {"expected", "label", "candidate", "decision", "logits", "text"}:
                    block(item)

    for directory in Path("/workspace/houndoracle-data").glob("campaign-*"):
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if not path.is_file() or path.stat().st_size > 60_000_000:
                continue
            if path.suffix == ".jsonl" and "token" not in path.name:
                for line in path.open():
                    try:
                        block(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                metadata_files += 1
            elif path.suffix == ".json":
                try:
                    block(json.loads(path.read_text()))
                except json.JSONDecodeError:
                    continue
                metadata_files += 1
    sources = {}
    rejected = Counter()
    if args.source == "vitaminc":
        pool = vitamin_pool(blocked_pages, blocked_claims, sources)
    else:
        original = {r["id"]: r for r in stream(URLS["original"], sources, "original")}
        accepted = {}
        pages = defaultdict(set)
        rejected = Counter()
        graph = Components()
        for r in stream(URLS["mirror"], sources, "mirror"):
            source = original.get(r.get("original_id"))
            if (
                not source
                or source.get("label") not in LABELS
                or r.get("label") != source["label"]
                or r.get("claim") != source["claim"]
            ):
                rejected["annotation_mismatch"] += 1
                continue
            es = r.get("evidence")
            if not isinstance(es, list) or any(
                not isinstance(e, list)
                or len(e) < 3
                or not isinstance(e[0], str)
                or not isinstance(e[2], str)
                for e in es
            ):
                rejected["evidence_schema"] += 1
                continue
            current_pages = {normalize(e[0]) for e in es}
            anchor = "case:fever-train:" + str(source["id"])
            for page in current_pages:
                graph.union(anchor, "page:" + page)
            graph.union(anchor, "claim:" + normalize(source["claim"]))
            if current_pages & blocked_pages or normalize(source["claim"]) in blocked_claims:
                rejected["prior_component_overlap"] += 1
                continue
            if source["label"] != "NOT ENOUGH INFO":
                supplied = {(e[0], int(e[1])) for e in es}
                gold_sets = [
                    {(e[2], e[3]) for e in group if len(e) > 3 and e[2] is not None}
                    for group in source.get("evidence", [])
                ]
                if not any(g and g <= supplied for g in gold_sets):
                    rejected["no_complete_original_evidence_set"] += 1
                    continue
            identity = "fever-train:" + str(source["id"])
            anchor = "case:" + identity
            for page in current_pages:
                graph.union(anchor, "page:" + page)
            graph.union(anchor, "claim:" + normalize(source["claim"]))
            pages[identity].update(current_pages)
            if identity not in accepted:
                accepted[identity] = (source, es)
        # Multiple evidence variants and shared articles form a single component.
        components = defaultdict(list)
        for identity, (source, es) in accepted.items():
            components[graph.root("case:" + identity)].append((identity, source, es))
        pool = []
        blocked_roots = {graph.root("page:" + page) for page in blocked_pages} | {
            graph.root("claim:" + claim) for claim in blocked_claims
        }
        for group, variants in components.items():
            if graph.root(group) in blocked_roots:
                rejected["transitive_prior_component_overlap"] += len(variants)
                continue
            identity, source, es = min(variants, key=lambda v: digest([2003, v[0]]))
            pool.append(
                {
                    "id": identity,
                    "group_id": digest([group]),
                    "question": source["claim"],
                    "expected": LABELS[source["label"]],
                    "source_ids": sorted("wiki:" + p for p in pages[identity])
                    or ["nei-case:" + identity],
                    "event_id": digest([group]),
                    "evidence": [
                        {
                            "url": "https://en.wikipedia.org/wiki/" + e[0],
                            "page": e[0],
                            "sentence_id": int(e[1]),
                            "text": e[2],
                        }
                        for e in es
                    ],
                    "label_origin": "human_dataset",
                    "synthetic": False,
                    "annotation_reference": URLS["original"] + "#" + str(source["id"]),
                    "label_basis": "passage_only_nei"
                    if source["label"] == "NOT ENOUGH INFO"
                    else "original_human_annotation",
                    "as_of": None,
                    "evidence_scope": "supplied_passages",
                    "audit_status": "pending",
                    "audit_reference": None,
                    "commercial_use_reviewed": False,
                    "license_reference": "https://fever.ai/download/fever/license.html",
                    "categories": [],
                    "domain": "unclassified",
                }
            )
    pool.sort(key=lambda r: digest([2003, r["group_id"]]))
    available = Counter(r["expected"] for r in pool)
    needs = {"development": 500, "validation": 500, "calibration": 1000, "locked": 1000}
    total = Counter({"YES": 1200, "NO": 1200, "MAYBE": 600})
    if any(available[k] < v for k, v in total.items()):
        (PUBLIC / "candidate-preflight.json").write_text(
            json.dumps(
                {
                    "status": "INSUFFICIENT_UNUSED_COMPONENTS",
                    "available": dict(available),
                    "required": dict(total),
                    "sources": sources,
                    "rejected": dict(rejected),
                },
                indent=2,
            )
            + "\n"
        )
        raise ValueError("Cannot shrink primary datasets silently")
    PRIVATE.mkdir()
    remaining = pool
    panels = {}
    reserved = set()
    for name, n in needs.items():
        wanted = {"YES": n * 4 // 10, "NO": n * 4 // 10, "MAYBE": n * 2 // 10}
        selected = []
        counts = Counter()
        for row in remaining:
            if counts[row["expected"]] < wanted[row["expected"]]:
                selected.append(row)
                counts[row["expected"]] += 1
                reserved.add(row["group_id"])
        remaining = [r for r in remaining if r["group_id"] not in reserved]
        target = PRIVATE / (name + ".jsonl")
        target.write_text("".join(json.dumps(r) + "\n" for r in selected))
        panels[name] = {
            "count": len(selected),
            "label_counts": dict(counts),
            "sha256": hashlib.file_digest(target.open("rb"), "sha256").hexdigest(),
        }
    training = remaining[:10000]
    target = PRIVATE / "training-reserve.jsonl"
    target.write_text("".join(json.dumps(r) + "\n" for r in training))
    panels["training-reserve"] = {
        "count": len(training),
        "sha256": hashlib.file_digest(target.open("rb"), "sha256").hexdigest(),
    }
    report = {
        "status": "RESERVED_UNSCORED_AUDITS_PENDING_NOT_QUALIFIED",
        "seed": 2003,
        "source_family": args.source,
        "sources": sources,
        "available_component_labels": dict(available),
        "rejected": dict(rejected),
        "panels": panels,
        "known_previous_training_and_evaluation_components_excluded": True,
        "metadata_files_checked": metadata_files,
        "sealed39_metadata_only_used_for_exclusion": True,
        "sealed39_labels_scored": False,
        "model_predictions_computed": 0,
        "independent_source_audits_completed": 0,
        "commercial_rights_review": "Pending; do not infer dataset rights from code license or redistribute text.",
        "remaining_work": "Source/temporal audit, subjectdomain/difficulty classification and separate endtoendgold. PassageNEI cannot be copied to webMAYBE. Reserved locked panels must not be scored or used in training.",
        "base_model_pretraining_contamination": "Unknown beyond known training/component exclusions; do not claim guaranteed absence.",
    }
    (PUBLIC / "candidate-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
