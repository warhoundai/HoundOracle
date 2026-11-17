"""Isolated repeated validation workloads for latency and exact local feature reuse."""

import json
import random
import resource
import time
from pathlib import Path

from houndoracle.experiments.head import HeadNLI
from houndoracle.experiments.metrics import paired_comparison, summary
from houndoracle.experiments.nli import OnnxNLI
from houndoracle.experiments.pipeline import ResearchPipeline

rows = [
    json.loads(s)
    for s in Path("/workspace/houndoracle-data/campaign-02/validation.jsonl")
    .read_text()
    .splitlines()
]
# Fixed balanced validation subset; no heldout-test-driven selection.
subset = []
for label in ["YES", "NO", "MAYBE"]:
    subset.extend([r for r in rows if r["expected"] == label][:20])
heads = json.loads(Path("research/experiments/results/trained-aggregators.json").read_text())
pipelines = {}
load_seconds = {}
for variant in ["fp32", "int8"]:
    name = "mlp" if variant == "fp32" else "mlp-int8"
    started = time.perf_counter()
    model = HeadNLI(
        OnnxNLI("/workspace/houndoracle-models/minilm", variant=variant), heads["mlp"], "mlp"
    )
    artifact = json.loads(Path(f"research/experiments/results/calibration-{name}.json").read_text())
    pipelines[variant] = ResearchPipeline(model, artifact)
    load_seconds[variant] = time.perf_counter() - started
cold = {v: [] for v in pipelines}
warm = {v: [] for v in pipelines}
rng = random.Random(1729)
for repeat in range(3):
    order = list(subset)
    rng.shuffle(order)
    for row in order:
        variants = list(pipelines)
        rng.shuffle(variants)
        for variant in variants:
            pipeline = pipelines[variant]
            result = pipeline.verify_evidence(
                row["question"],
                row["evidence"],
                population=pipeline.artifact["population"],
                use_cache=False,
            )
            item = {
                "id": row["id"] + ":" + str(repeat),
                "group_id": row["group_id"],
                "expected": row["expected"],
                "decision": result["decision"],
                "latency_seconds": result["latency_seconds"],
                "cache_hit": False,
                "all_in_cost_usd": None,
            }
            cold[variant].append(item)
            # Priming call recorded separately from the measured warm-cache hit.
            pipeline.verify_evidence(
                row["question"], row["evidence"], population=pipeline.artifact["population"]
            )
            reused = pipeline.verify_evidence(
                row["question"], row["evidence"], population=pipeline.artifact["population"]
            )
            assert reused["decision"] == result["decision"] and reused["cache_hit"]
            warm[variant].append(
                {**item, "latency_seconds": reused["latency_seconds"], "cache_hit": True}
            )
report = {
    "workload": "60 fixed validation questions x 3 interleaved repetitions; evidence already supplied",
    "model_load_seconds": load_seconds,
    "peak_combined_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "cold": {v: summary(r) for v, r in cold.items()},
    "warm": {v: summary(r) for v, r in warm.items()},
    "paired_quantization": paired_comparison(cold["fp32"], cold["int8"]),
    "limitations": [
        "Both model variants resident; peak RSS is not per-model memory.",
        "Cold means no answer/feature cache, not a new process or downloaded model.",
        "Warm priming is excluded from hit latency; cache-hit rates here are workload construction, not production forecasts.",
        "No S3, live searches, Gemini agent calls or invoiced costs were measured.",
    ],
}
Path("benchmarks/results/campaign-02/reuse-profile.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(
    json.dumps(
        {k: v for k, v in report.items() if k in ["cold", "warm", "paired_quantization"]}, indent=2
    )
)
