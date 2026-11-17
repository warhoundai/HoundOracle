"""Measure the frozen trained verifier on the exact new Gemini cohort before remote inference."""

import hashlib
import json
import random
import time
from pathlib import Path

from houndoracle.experiments.head import HeadNLI
from houndoracle.experiments.metrics import summary
from houndoracle.experiments.nli import OnnxNLI
from houndoracle.experiments.pipeline import ResearchPipeline


def main():
    safe = Path("benchmarks/results/campaign-14")
    target = safe / "local-records.jsonl"
    if target.exists():
        raise FileExistsError("Local comparison already frozen")
    payload = json.loads(Path("/workspace/houndoracle-data/public-gemini-payload.json").read_text())
    artifact = json.loads(
        Path("research/experiments/campaign-09/continued-mixture-calibration.json").read_text()
    )
    manifest = json.loads(
        Path("research/experiments/campaign-08/round-2/trained-manifest.json").read_text()
    )
    head = json.loads(
        Path("research/experiments/campaign-09/continued-mixture-head.json").read_text()
    )
    started = time.perf_counter()
    model = ResearchPipeline(
        HeadNLI(
            OnnxNLI(
                "/workspace/houndoracle-models/minilm-vitc-fever-1921",
                variant="fp32",
                threads=1,
                max_length=256,
                reviewed_manifest=manifest,
            ),
            head,
            "continued-mixture",
        ),
        artifact,
    )
    initialization = time.perf_counter() - started
    warm = [
        json.loads(line)
        for line in Path("/workspace/houndoracle-data/campaign-08/vitaminc-train.jsonl")
        .read_text()
        .splitlines()
    ][:10]
    for row in warm:
        model.model.predict(row["question"], row["evidence"])
    tasks = [
        (repeat, row) for repeat in range(payload["repetitions"]) for row in payload["questions"]
    ]
    random.Random(1930).shuffle(tasks)
    records = []
    for repeat, row in tasks:
        start = time.perf_counter()
        result = model.verify_evidence(
            row["question"], row["evidence"], population=artifact["population"], use_cache=False
        )
        records.append(
            {k: row[k] for k in ["id", "group_id", "expected"]}
            | {
                k: result.get(k)
                for k in [
                    "decision",
                    "candidate",
                    "confidence",
                    "raw_score",
                    "truncated",
                    "cache_hit",
                ]
            }
            | {
                "repeat": repeat,
                "latency_seconds": time.perf_counter() - start,
                "all_in_cost_usd": None,
                "generative_model_calls": 0,
                "local_neural_calls": 1,
                "search_api_calls": 0,
                "citation_supported": None,
            }
        )
    target.write_text("".join(json.dumps(r) + "\n" for r in records))
    report = summary(records) | {
        "unique_queries": len(payload["questions"]),
        "repetitions": payload["repetitions"],
        "model_initialization_seconds": initialization,
        "model_config": model.model.config,
        "dataset_sha256": hashlib.sha256(
            json.dumps(payload["questions"], sort_keys=True).encode()
        ).hexdigest(),
        "scope": "Standalone local conditional verifier, one thread, warm model, result cache disabled; not client-to-cloud or agent/hybrid latency.",
    }
    (safe / "local-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
