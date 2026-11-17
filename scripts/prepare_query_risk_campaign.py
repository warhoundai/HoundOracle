"""Leakage-aware public revision pairs; source text and token inputs stay outside the repository."""

import hashlib
import json
import unicodedata
from pathlib import Path
from urllib.parse import unquote

from prepare_abstention_data import Components

from houndoracle.experiments.dataset import digest, load_checked


def norm(value):
    return " ".join(
        unicodedata.normalize("NFC", unquote(value)).replace("_", " ").casefold().split()
    )


def text_key(value):
    return hashlib.sha256(norm(value).encode()).hexdigest()


def main():
    target = Path("/workspace/houndoracle-data/campaign-09")
    if target.exists():
        raise FileExistsError("Preserve sealed dataset")
    graph = Components()
    rows = []
    exclusions = {}
    ids = set()
    for part in ["train", "dev", "test"]:
        with Path("/workspace/houndoracle-data/vitaminc-sources", part + ".jsonl").open() as source:
            for line in source:
                r = json.loads(line)
                if r["revision_type"] != "real" or r.get("FEVER_id"):
                    exclusions["synthetic_or_fever_derived"] = (
                        exclusions.get("synthetic_or_fever_derived", 0) + 1
                    )
                    continue
                if not all(
                    isinstance(r.get(k), str) and r[k].strip()
                    for k in ["unique_id", "case_id", "claim", "evidence", "page"]
                ):
                    exclusions["invalid_or_empty"] = exclusions.get("invalid_or_empty", 0) + 1
                    continue
                if r["unique_id"] in ids:
                    raise ValueError("Duplicate public example ID")
                ids.add(r["unique_id"])
                anchor = "page:" + norm(r["page"])
                for key in [
                    "case:" + r["case_id"],
                    "claim:" + text_key(r["claim"]),
                    "span:" + text_key(r["evidence"]),
                ]:
                    graph.union(anchor, key)
                rows.append(r | {"part": part, "anchor": anchor})
    old, _ = load_checked("/workspace/houndoracle-data/fever-sources")
    seen = old.copy()
    for pattern in [
        "campaign-06/*.jsonl",
        "campaign-06-balanced/*.jsonl",
        "campaign-06-successor/*.jsonl",
    ]:
        for path in Path("/workspace/houndoracle-data").glob(pattern):
            seen.extend(json.loads(s) for s in path.read_text().splitlines())
    blocked = set()
    for r in seen:
        for page in r["source_pages"]:
            blocked.add(graph.root("page:" + norm(page)))
        blocked.add(graph.root("claim:" + text_key(r["question"])))
        for item in r["evidence"]:
            blocked.add(graph.root("span:" + text_key(item["text"])))
    groups = {}
    for r in rows:
        groups.setdefault(graph.root(r["anchor"]), []).append(r)
    pools = {"train": [], "dev": [], "test": []}
    for component, group in groups.items():
        parts = {r["part"] for r in group}
        if len(parts) != 1:
            exclusions["cross_official_split_rows"] = exclusions.get(
                "cross_official_split_rows", 0
            ) + len(group)
            continue
        part = next(iter(parts))
        if component in blocked:
            exclusions["previously_observed_eval_rows"] = exclusions.get(
                "previously_observed_eval_rows", 0
            ) + len(group)
            continue
        cases = {}
        for r in group:
            cases.setdefault(r["case_id"], []).append(r)
        eligible = [
            case
            for case in cases.values()
            if len(case) >= 2 and all(len(r["claim"]) + len(r["evidence"]) < 5000 for r in case)
        ]
        if not eligible:
            continue
        chosen = min(eligible, key=lambda case: digest([1920, case[0]["case_id"]]))
        gid = digest(sorted(r["unique_id"] for r in group))
        pools[part].append((gid, chosen))
    for values in pools.values():
        values.sort(key=lambda item: digest([1920, item[0]]))

    def take(part, n):
        chosen = []
        while len(chosen) < n and pools[part]:
            gid, case = pools[part].pop(0)
            for r in case:
                chosen.append(
                    {
                        "id": "vitaminc:" + r["unique_id"],
                        "group_id": gid,
                        "case_id": r["case_id"],
                        "question": r["claim"],
                        "expected": {
                            "SUPPORTS": "YES",
                            "REFUTES": "NO",
                            "NOT ENOUGH INFO": "MAYBE",
                        }[r["label"]],
                        "category": "contrastive-real-revision",
                        "source_pages": [r["page"]],
                        "evidence": [
                            {
                                "page": r["page"],
                                "text": r["evidence"],
                                "url": "https://en.wikipedia.org/wiki/"
                                + r["page"].replace(" ", "_"),
                                "recorded_revision_id": r["wiki_revision_id"],
                            }
                        ],
                        "benchmark_scope": "conditional on supplied revision evidence; not current-world truth",
                    }
                )
        if len(chosen) < n:
            raise ValueError(f"Insufficient independent {part} pairs: {len(chosen)}")
        return chosen

    used = set()
    for directory in [
        Path("/workspace/houndoracle-data/campaign-08"),
        Path("/workspace/houndoracle-data/campaign-08-round2"),
        Path("/workspace/houndoracle-data/campaign-08-round2-power"),
    ]:
        for path in directory.glob("*.jsonl"):
            if path.name in ["training-token-inputs.jsonl"]:
                continue
            for row in map(json.loads, path.read_text().splitlines()):
                if isinstance(row, dict) and "group_id" in row:
                    used.add(row["group_id"])
    for part, values in pools.items():
        pools[part] = [(g, case) for g, case in values if g not in used]
        pools[part].sort(key=lambda item: digest([1922, item[0]]))

    def convert(r, gid):
        return {
            "id": "vitaminc:" + r["unique_id"],
            "group_id": gid,
            "case_id": r["case_id"],
            "question": r["claim"],
            "expected": {"SUPPORTS": "YES", "REFUTES": "NO", "NOT ENOUGH INFO": "MAYBE"}[
                r["label"]
            ],
            "category": "independent-revision-query",
            "source_pages": [r["page"]],
            "evidence": [
                {
                    "page": r["page"],
                    "text": r["evidence"],
                    "url": "https://en.wikipedia.org/wiki/" + r["page"].replace(" ", "_"),
                }
            ],
            "benchmark_scope": "Component-uniform, then case-query-uniform conditional evidence label; not current-world truth",
        }

    def queries(part, n):
        chosen = []
        paired = []
        for gid, case in pools[part][:n]:
            row = min(case, key=lambda r: digest([1922, r["unique_id"]]))
            chosen.append(convert(row, gid))
            paired.extend(convert(r, gid) for r in case)
        if len(chosen) != n:
            raise ValueError("Insufficient independent queries")
        pools[part] = pools[part][n:]
        return chosen, paired

    fitting, _ = queries("train", 1000)
    risk, _ = queries("train", 1000)
    locked, paired = queries("test", 600)
    oldval = [
        json.loads(line)
        for line in Path("/workspace/houndoracle-data/campaign-08/validation.jsonl")
        .read_text()
        .splitlines()
    ]
    vgroups = {}
    for row in oldval:
        vgroups.setdefault(row["group_id"], []).append(row)
    validation = [min(group, key=lambda r: digest([1922, r["id"]])) for group in vgroups.values()]
    panels = {
        "calibration-fit": fitting,
        "calibration-risk": risk,
        "validation": validation,
        "locked": locked,
        "locked-pairs": paired,
    }
    target.mkdir()
    manifest = {
        "seed": 1922,
        "sampling": "One query per independent component; conditional population is uniform components then hash-selected query in a complete contrastive case. Not the old pair-weighted/case-any-error estimand.",
        "panels": {},
        "remaining_components": {k: len(v) for k, v in pools.items()},
    }
    for name, selected in panels.items():
        path = target / (name + ".jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in selected))
        manifest["panels"][name] = {
            "n": len(selected),
            "groups": len({r["group_id"] for r in selected}),
            "class_counts": {
                y: sum(r["expected"] == y for r in selected) for y in ["YES", "NO", "MAYBE"]
            },
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "ids": [r["id"] for r in selected],
        }
    for a, b in [
        ("calibration-fit", "calibration-risk"),
        ("calibration-fit", "locked"),
        ("calibration-risk", "locked"),
        ("validation", "locked"),
    ]:
        if {r["group_id"] for r in panels[a]} & {r["group_id"] for r in panels[b]}:
            raise ValueError("Independent query component overlap")
    Path("research/experiments/campaign-09/dataset-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: {a: b for a, b in v.items() if a != "ids"} for k, v in manifest["panels"].items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
