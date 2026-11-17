"""A separately declared balanced panel: fact/entity disjoint, not whole-page disjoint."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from prepare_abstention_data import Components

from houndoracle.experiments.dataset import digest, load_checked


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def fact_keys(row):
    keys = ["claim:" + normalize(row["question"]), "subject:" + normalize(row["subject"])]
    question = normalize(row["question"])
    for evidence in row["evidence"]:
        keys += [
            "fact:" + normalize(evidence["page"]) + ":" + str(evidence["sentence_id"]),
            "span:" + hashlib.sha256(normalize(evidence["text"]).encode()).hexdigest(),
        ]
        title = evidence["page"].split("_-LRB-")[0].replace("_", " ")
        if re.match(re.escape(normalize(title)) + r"(?:\b|$)", question):
            keys.append("primary-entity:" + normalize(title))
    return keys


def main():
    import sys

    extend = "--extend-risk" in sys.argv
    target = Path("/workspace/houndoracle-data/campaign-06-balanced")
    if target.exists() and not extend:
        raise FileExistsError("Balanced panel already sealed")
    old, _ = load_checked("/workspace/houndoracle-data/fever-sources")
    rows, excluded = load_checked("/workspace/houndoracle-data/fever-train-sources")
    c = Components()
    for row in old + rows:
        for key in fact_keys(row):
            c.union("id:" + row["id"], key)
    forbidden = {c.root("id:" + r["id"]) for r in old}
    groups = {}
    for row in rows:
        root = c.root("id:" + row["id"])
        if root not in forbidden:
            groups.setdefault(root, []).append(row)
    ordered = sorted(groups.values(), key=lambda g: digest([1913, sorted(r["id"] for r in g)]))
    used = set()
    if extend:
        if (target / "risk-extended.jsonl").exists():
            raise FileExistsError("Extended risk set already frozen")
        for name in ["risk", "locked"]:
            used.update(
                json.loads(line)["group_id"]
                for line in (target / (name + ".jsonl")).read_text().splitlines()
            )
        selected = []
        for label in ["YES", "NO", "MAYBE"]:
            count = 0
            for group in ordered:
                gid = digest(sorted(r["id"] for r in group))
                if gid in used:
                    continue
                row = next(
                    (
                        r
                        for r in sorted(group, key=lambda r: digest([1913, r["id"]]))
                        if r["expected"] == label
                        and sum(len(e["text"]) for e in r["evidence"]) < 5000
                    ),
                    None,
                )
                if row:
                    selected.append(row | {"group_id": gid})
                    used.add(gid)
                    count += 1
                if count == 200:
                    break
            if count != 200:
                raise ValueError("Insufficient unassigned groups for risk power extension")
        previous = [json.loads(s) for s in (target / "risk.jsonl").read_text().splitlines()]
        path = target / "risk-extended.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in previous + selected))
        record = {
            "rows": len(previous + selected),
            "added_rows": len(selected),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "reason": "Power extension chosen from old-validation false-abstention rate before any fresh model predictions; original risk and locked files preserved.",
            "class_counts": {
                v: sum(r["expected"] == v for r in previous + selected)
                for v in ["YES", "NO", "MAYBE"]
            },
        }
        Path("research/experiments/campaign-06/risk-power-extension.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        print(json.dumps(record, indent=2))
        return
    splits = {}
    for split, count in [("risk", 200), ("locked", 100)]:
        selected = []
        for label in ["YES", "NO", "MAYBE"]:
            n = 0
            for group in ordered:
                gid = digest(sorted(r["id"] for r in group))
                if gid in used:
                    continue
                row = next(
                    (
                        r
                        for r in sorted(group, key=lambda r: digest([1913, r["id"]]))
                        if r["expected"] == label
                        and sum(len(e["text"]) for e in r["evidence"]) < 5000
                    ),
                    None,
                )
                if row is not None:
                    selected.append(row | {"group_id": gid})
                    used.add(gid)
                    n += 1
                if n == count:
                    break
            if n != count:
                raise ValueError(f"Insufficient independent groups for {split}/{label}: {n}")
        splits[split] = selected
    prior = {key for row in old for key in fact_keys(row)}
    risk = {key for row in splits["risk"] for key in fact_keys(row)}
    test = {key for row in splits["locked"] for key in fact_keys(row)}
    assert not risk & test and not risk & prior and not test & prior
    target.mkdir()
    record = {
        "regime": "fact-span/leading-subject/primary-entity components; whole pages MAY overlap",
        "reason": "Whole-page transitive grouping left a ~90% MAYBE residual; preserve that regime separately and add a balanced task panel before scoring, without prediction filtering.",
        "seed": 1913,
        "eligible_components": len(groups),
        "source_exclusions": excluded,
        "direct_fact_entity_claim_overlap_old_risk_test": 0,
        "splits": {},
        "limitations": "Unknown aliases/paraphrases and shared background articles may still correlate errors. This is not the earlier stricter whole-page-disjoint benchmark and does not prove universal independence.",
    }
    for split, selected in splits.items():
        path = target / (split + ".jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in selected))
        record["splits"][split] = {
            "rows": len(selected),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "class_counts": {
                v: sum(r["expected"] == v for r in selected) for v in ["YES", "NO", "MAYBE"]
            },
            "ids": [r["id"] for r in selected],
        }
    record["shared_page_count_risk_locked"] = len(
        {p for r in splits["risk"] for p in r["source_pages"]}
        & {p for r in splits["locked"] for p in r["source_pages"]}
    )
    Path("research/experiments/campaign-06/balanced-manifest.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in record.items() if k != "splits"}))
    print({s: len(r) for s, r in splits.items()})


if __name__ == "__main__":
    main()
