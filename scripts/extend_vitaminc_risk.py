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
    target = Path("/workspace/houndoracle-data/campaign-08-round2-power")
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
        if part != "train" and component in blocked:
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

    parent = Path("/workspace/houndoracle-data/campaign-08")
    second = Path("/workspace/houndoracle-data/campaign-08-round2")
    used = set()
    for directory in [parent, second]:
        for name in [
            "development",
            "vitaminc-train",
            "validation",
            "calibration-fit",
            "calibration-risk",
            "locked",
        ]:
            for row in map(json.loads, (directory / (name + ".jsonl")).read_text().splitlines()):
                used.add(row["group_id"])
    for part, values in pools.items():
        pools[part] = [(g, case) for g, case in values if g not in used]
    extra = take("dev", 1200)
    original = [
        json.loads(line) for line in (second / "calibration-risk.jsonl").read_text().splitlines()
    ]
    combined = original + extra
    target.mkdir()
    path = target / "calibration-risk.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in combined))
    public = Path("research/experiments/campaign-08/round-2")
    manifest = json.loads((public / "dataset-manifest.json").read_text())
    manifest["panels"]["calibration-risk"] = {
        "n": len(combined),
        "groups": len({r["group_id"] for r in combined}),
        "class_counts": {
            y: sum(r["expected"] == y for r in combined) for y in ["YES", "NO", "MAYBE"]
        },
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "ids": [r["id"] for r in combined],
    }
    manifest["risk_source"] = str(path)
    manifest["power_amendment"] = {
        "before_round2_predictions": True,
        "added_pairs": len(extra),
        "purpose": "More independent components to distinguish case-any-error risk from finite-sample uncertainty; keep original600 file and all thresholds unchanged.",
    }
    (public / "dataset-manifest-extended.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "risk_pairs": len(combined),
                "risk_groups": len({r["group_id"] for r in combined}),
                "added": len(extra),
            }
        )
    )


if __name__ == "__main__":
    main()
