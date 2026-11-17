"""Freeze an unfunded fresh cohort. Preparation performs no cloud or model calls."""

import hashlib
import json
from pathlib import Path

from houndoracle.experiments.dataset import digest, load_checked


def main():
    rows, _ = load_checked("/workspace/houndoracle-data/fever-sources")
    parents = {}

    def find(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]

    for r in rows:
        for key in ["page:" + p for p in r["source_pages"]] + [
            "subject:" + r["subject"],
            "claim:" + r["question"].casefold(),
        ]:
            parents[find(r["id"])] = find(key)
    groups = {}
    for r in rows:
        groups.setdefault(find(r["id"]), []).append(r)
    forbidden = set()
    for directory in ["campaign-02", "scores-invalidated-v1"]:
        for p in Path("/workspace/houndoracle-data", directory).glob("*.jsonl"):
            forbidden.update(json.loads(s)["id"] for s in p.read_text().splitlines())
    for path in Path("/workspace/houndoracle-data").glob("cloud-comparison-payload*.json"):
        p = json.loads(path.read_text())
        forbidden.update(r["id"] for part in ["development", "confirmation"] for r in p[part])
    available = [g for g in groups.values() if not any(r["id"] in forbidden for r in g)]
    available.sort(key=lambda g: digest([1907, sorted(r["id"] for r in g)]))
    selected = []
    used = set()
    # Select by declared labels/size only, never by model predictions or acceptance.
    for label in ["YES", "NO", "MAYBE"]:
        count = 0
        for group in available:
            gid = digest(sorted(r["id"] for r in group))
            if gid in used:
                continue
            row = next(
                (
                    r
                    for r in sorted(group, key=lambda r: digest([1907, r["id"]]))
                    if r["expected"] == label and sum(len(e["text"]) for e in r["evidence"]) <= 5000
                ),
                None,
            )
            if row:
                selected.append(
                    {k: row[k] for k in ["id", "question", "expected", "evidence"]}
                    | {"group_id": gid}
                )
                used.add(gid)
                count += 1
            if count == 16:
                break
    target = Path("/workspace/houndoracle-data/campaign-05-fresh.jsonl")
    if target.exists():
        raise FileExistsError("Fresh cohort already frozen; do not overwrite")
    target.write_text("".join(json.dumps(r) + "\n" for r in selected))
    n = len(selected)
    calls = 6 * n  # two live systems, three paired repetitions
    report = {
        "status": "PREPARED_NOT_AUTHORIZED_NOT_EXECUTED",
        "questions": n,
        "source_disjoint_groups": len(used),
        "available_groups_before_selection": len(available),
        "class_counts": {
            v: sum(r["expected"] == v for r in selected) for v in ["YES", "NO", "MAYBE"]
        },
        "ids": [r["id"] for r in selected],
        "dataset_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "maximum_model_calls": calls,
        "maximum_reserved_model_usd": calls * 0.021504,
        "proposed_compute_allowance_usd": 0.5,
        "authorization_requested_usd": round(calls * 0.021504 + 0.5 + 0.5, 2),
        "model": "gemini-3.8-flash",
        "sdk": "2.34.0",
        "thinking_level": "LOW",
        "settings": {
            "temperature": 0,
            "seed": 1907,
            "max_input_tokens": 8192,
            "max_output_tokens": 4096,
            "region": "global",
        },
        "controls": "Same evidence, model configuration, empty-corpus preflight and output format in both arms. Separate cold/warm conditions; no fabricated Gemini latency.",
        "decision_rule": "Freeze candidate after local validation; no holdout tuning. Accuracy/coverage noninferiority requires adequately powered paired intervals, otherwise inconclusive. Report category failures, not only aggregate gains.",
        "no_grounding": True,
        "no_gpu": True,
        "no_public_endpoint": True,
        "limitations": "Historical FEVER only. Does not establish benefits across current documentation, numerical, adversarial, retrieval, S3 or full-agent benchmarks. Model pretraining exposure unknown.",
    }
    Path("research/experiments/campaign-04/next-cloud-preflight.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
