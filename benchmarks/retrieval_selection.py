"""Paired context-packing diagnostic on reused historical supplied-corpus queries."""

import json
import time
from pathlib import Path

from houndoracle import Oracle, Store
from houndoracle.experiments.metrics import paired_comparison, summary
from houndoracle.experiments.nli import OnnxNLI
from houndoracle.models import DocumentInput


def main():
    target = Path("research/experiments/campaign-31/result.json")
    if target.exists():
        raise FileExistsError("Preserve completed paired experiment")
    rows = [
        json.loads(s)
        for s in Path("/workspace/houndoracle-data/campaign-09/validation.jsonl")
        .read_text()
        .splitlines()
    ]
    store = Store()
    oracle = Oracle(store)
    now = time.time()
    for row in rows:
        for e in row["evidence"]:
            store.ingest(
                DocumentInput(
                    url=e["url"],
                    text=e["text"],
                    origin="historical-snapshot",
                    observed_at=now,
                    expires_at=now + 3600,
                    reviewed=False,
                    primary=False,
                )
            )
    manifest = json.loads(
        Path("research/experiments/campaign-08/round-2/trained-manifest.json").read_text()
    )
    model = OnnxNLI(
        "/workspace/houndoracle-models/minilm-vitc-fever-1921",
        variant="fp32",
        threads=1,
        max_length=256,
        reviewed_manifest=manifest,
    )
    model.predict("A is B", [{"page": "warmup", "text": "A is B."}])
    results = {"top1": [], "top3": []}
    for i, row in enumerate(rows):
        hits = oracle.search(row["question"], 10)
        for k in [1, 3] if i % 2 else [3, 1]:
            then = time.perf_counter()
            evidence = [{"page": h["data"]["url"], "text": h["data"]["text"]} for h in hits[:k]]
            pred = model.predict(row["question"], evidence)
            results["top" + str(k)].append(
                {key: row[key] for key in ["id", "group_id", "expected"]}
                | {
                    "decision": pred["candidate"],
                    "latency_seconds": time.perf_counter() - then,
                    "context_truncated": pred["truncated"],
                    "citation_supported": None,
                    "all_in_cost_usd": None,
                }
            )
        if (i + 1) % 50 == 0:
            print(json.dumps({"scored": i + 1}), flush=True)
    a, b = summary(results["top3"]), summary(results["top1"])
    delta = b["accuracy_all_decisions"] - a["accuracy_all_decisions"]
    report = {
        "baseline_top3": a,
        "candidate_top1": b,
        "comparison": paired_comparison(
            results["top3"], results["top1"], seed=1964, replicates=2000
        ),
        "raw_accuracy_delta": delta,
        "research_gate_passed": bool(
            delta >= 0.02 and b["latency_seconds"]["p50"] <= a["latency_seconds"]["p50"]
        ),
        "production_qualified": False,
        "scope": "Same supplied-corpus reused development queries, raw encoder candidate labels. Selection latency is model/packing only, not agent workflow. Single-passage result does not justify dropping contradiction or multi-hop checks.",
        "records": results,
    }
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    store.close()


if __name__ == "__main__":
    main()
