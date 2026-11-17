"""Frozen-test disagreement diagnostics only; never chooses a new threshold or route."""

import json
import re
from collections import Counter
from pathlib import Path


def main():
    root = Path("benchmarks/results/campaign-14")
    local = {
        (r["id"], r["repeat"]): r
        for r in map(json.loads, (root / "local-records.jsonl").read_text().splitlines())
    }
    remote = list(map(json.loads, (root / "gemini-records.jsonl").read_text().splitlines()))
    payload = json.loads(Path("/workspace/houndoracle-data/public-gemini-payload.json").read_text())
    by_id = {r["id"]: r for r in payload["questions"]}
    counts = Counter()
    strata = {}
    rows = []
    decisions = {}
    for gemini in remote:
        oracle = local[(gemini["id"], gemini["repeat"])]
        g = not gemini.get("failure") and gemini["decision"] == gemini["expected"]
        o = oracle["candidate"] == oracle["expected"]
        key = (
            "both_correct"
            if g and o
            else "both_wrong"
            if not g and not o
            else "oracle_raw_only_correct"
            if o
            else "gemini_only_correct"
        )
        counts[key] += 1
        question = by_id[gemini["id"]]["question"]
        tags = [
            "numeric_tokens" if re.search(r"\d", question) else "no_numeric_tokens",
            "explicit_negation"
            if re.search(r"\b(not|never|no|neither|without)\b", question, re.IGNORECASE)
            else "no_explicit_negation",
            "gold_" + gemini["expected"],
        ]
        for tag in tags:
            entry = strata.setdefault(tag, Counter())
            entry["count"] += 1
            entry["gemini_correct"] += g
            entry["oracle_raw_correct"] += o
            entry["oracle_accepted"] += oracle["decision"] in ["YES", "NO"]
        decisions.setdefault(gemini["id"], set()).add(gemini["decision"])
        rows.append(
            {
                "id": gemini["id"],
                "repeat": gemini["repeat"],
                "group_id": gemini["group_id"],
                "expected": gemini["expected"],
                "gemini": gemini["decision"],
                "oracle_raw": oracle["candidate"],
                "oracle_calibrated": oracle["decision"],
                "diagnostic": key,
                "tags": tags,
            }
        )
    output = {
        "paired_slots": len(remote),
        "unique_components": len(by_id),
        "raw_verdict_disagreements": dict(counts),
        "strata": {k: dict(v) for k, v in strata.items()},
        "questions_with_gemini_decision_variation": sum(len(v) > 1 for v in decisions.values()),
        "scope": "Post-frozen-test diagnostics only. Raw Oracle candidates are not accepted answers. Strata are overlapping regex descriptors, not independently adjudicated task families. Three repetitions are not three independent ground-truth samples. No tuning on this cohort follows.",
    }
    (root / "failure-analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    (root / "disagreements.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
