"""New untouched panels after a failed heldout test; old outcomes are explicitly retired."""

import hashlib
import json
from pathlib import Path

from prepare_abstention_data import Components, keys
from prepare_balanced_fact_panel import fact_keys

from houndoracle.experiments.dataset import digest, load_checked


def main():
    target = Path("/workspace/houndoracle-data/campaign-06-successor")
    if target.exists():
        raise FileExistsError("Successor panels already sealed")
    old, _ = load_checked("/workspace/houndoracle-data/fever-sources")
    rows, _ = load_checked("/workspace/houndoracle-data/fever-train-sources")
    additional, _ = load_checked("/workspace/houndoracle-data/fever-test-sources")
    rows += additional
    seen = []
    for name in [
        "/workspace/houndoracle-data/campaign-06/risk.jsonl",
        "/workspace/houndoracle-data/campaign-06/locked.jsonl",
        "/workspace/houndoracle-data/campaign-06-balanced/risk-extended.jsonl",
        "/workspace/houndoracle-data/campaign-06-balanced/locked.jsonl",
    ]:
        seen += [json.loads(s) for s in Path(name).read_text().splitlines()]
    pages = Components()
    facts = Components()
    for row in old + rows:
        for key in keys(row):
            pages.union("id:" + row["id"], key)
        for key in fact_keys(row):
            facts.union("id:" + row["id"], key)
    blocked_pages = {pages.root("id:" + r["id"]) for r in old + seen}
    blocked_facts = {facts.root("id:" + r["id"]) for r in old + seen}
    page_groups = {}
    fact_groups = {}
    for row in rows:
        pg = pages.root("id:" + row["id"])
        fg = facts.root("id:" + row["id"])
        if pg not in blocked_pages:
            page_groups.setdefault(pg, []).append(row)
        if fg not in blocked_facts:
            fact_groups.setdefault(fg, []).append(row)
    ordered_pages = sorted(
        page_groups.items(), key=lambda item: digest([1915, sorted(r["id"] for r in item[1])])
    )
    strict = []
    selected_page_roots = set()
    for root, group in ordered_pages:
        row = next(
            (
                r
                for r in sorted(group, key=lambda r: digest([1915, r["id"]]))
                if sum(len(e["text"]) for e in r["evidence"]) < 5000
            ),
            None,
        )
        if row:
            strict.append(row | {"group_id": digest(sorted(r["id"] for r in group))})
            selected_page_roots.add(root)
        if len(strict) == 600:
            break
    if len(strict) != 600:
        raise ValueError(f"Only {len(strict)} unused strict components remain")
    cross_blocked = {facts.root("id:" + r["id"]) for r in strict}
    ordered_facts = sorted(
        [(root, g) for root, g in fact_groups.items() if root not in cross_blocked],
        key=lambda item: digest([1915, sorted(r["id"] for r in item[1])]),
    )
    used = set()
    panels = {"strict-risk": strict[:400], "strict-locked": strict[400:]}
    for split, count in [("balanced-risk", 100), ("balanced-locked", 100)]:
        selected = []
        for label in ["YES", "NO", "MAYBE"]:
            n = 0
            for root, group in ordered_facts:
                if root in used:
                    continue
                row = next(
                    (
                        r
                        for r in sorted(group, key=lambda r: digest([1915, r["id"]]))
                        if r["expected"] == label
                        and sum(len(e["text"]) for e in r["evidence"]) < 5000
                    ),
                    None,
                )
                if row:
                    selected.append(row | {"group_id": digest(sorted(r["id"] for r in group))})
                    used.add(root)
                    n += 1
                if n == count:
                    break
            if n != count:
                raise ValueError(f"Insufficient successor {split}/{label}: {n}")
        panels[split] = selected
    target.mkdir()
    manifest = {
        "seed": 1915,
        "retired_unique_rows": len({r["id"] for r in seen}),
        "regimes": "Strict whole-page disjoint; balanced fact/subject/entity disjoint with possible shared background pages. Cross-regime successor cases excluded by fact/entity components; background pages may overlap across regimes, so results must not be pooled as independent.",
        "selection": "No model scores used; no retired or original components admitted under each regime.",
        "panels": {},
    }
    for name, selected in panels.items():
        path = target / (name + ".jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in selected))
        manifest["panels"][name] = {
            "rows": len(selected),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "class_counts": {
                v: sum(r["expected"] == v for r in selected) for v in ["YES", "NO", "MAYBE"]
            },
            "ids": [r["id"] for r in selected],
        }
    Path("research/experiments/campaign-06/successor-manifest.json").write_text(
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
