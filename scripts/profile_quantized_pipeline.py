"""Isolated local SDK-path timing and process memory; never a Gemini/network speed claim."""

import argparse
import json
import random
import resource
import time
from pathlib import Path

import numpy as np
from quantization_common import campaign_index
from score_query_risk import MODELS

from houndoracle.experiments.head import HeadNLI
from houndoracle.experiments.nli import FILES, OnnxNLI
from houndoracle.experiments.pipeline import ResearchPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=["fp32"] + [n for n in MODELS if n.startswith("int8")])
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--mixed", action="store_true")
    args = parser.parse_args()
    number = campaign_index(args)
    public = Path(f"research/experiments/campaign-{number:02d}")
    safe = Path(f"benchmarks/results/campaign-{number:02d}")
    artifact = json.loads((public / (args.variant + "-calibration.json")).read_text())
    head = json.loads((public / (args.variant + "-head.json")).read_text())
    directory, manifest_path = MODELS[args.variant]
    manifest = json.loads(Path(manifest_path).read_text())
    variant = "fp32" if args.variant == "fp32" else "int8"
    started = time.perf_counter()
    model = OnnxNLI(
        "/workspace/houndoracle-models/" + directory,
        variant=variant,
        threads=1,
        max_length=256,
        reviewed_manifest=manifest,
    )
    pipeline = ResearchPipeline(HeadNLI(model, head, args.variant), artifact)
    load_seconds = time.perf_counter() - started
    rows = [
        json.loads(line)
        for line in Path(f"/workspace/houndoracle-data/campaign-{number:02d}/validation.jsonl")
        .read_text()
        .splitlines()
    ]
    for row in rows[:10]:
        pipeline.verify_evidence(
            row["question"], row["evidence"], population=artifact["population"], use_cache=False
        )
    records = []
    rng = random.Random(1923)
    for repeat in range(3):
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        for i in indices:
            row = rows[i]
            start = time.perf_counter()
            result = pipeline.verify_evidence(
                row["question"], row["evidence"], population=artifact["population"], use_cache=False
            )
            records.append(
                {
                    "id": row["id"],
                    "repeat": repeat,
                    "seconds": time.perf_counter() - start,
                    "decision": result["decision"],
                    "cache_hit": result["cache_hit"],
                }
            )
    rss = None
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
    durations = [r["seconds"] for r in records]
    summary = {
        "variant": args.variant,
        "queries": len(records),
        "unique_queries": len(rows),
        "repetitions": 3,
        "threads": 1,
        "model_initialization_seconds": load_seconds,
        "p50_seconds": float(np.quantile(durations, 0.5)),
        "p95_seconds": float(np.quantile(durations, 0.95)),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "resident_bytes_after_queries": rss,
        "model_bytes": Path("/workspace/houndoracle-models", directory, FILES[variant])
        .stat()
        .st_size,
        "scope": "Isolated process, warm model, local verify_evidence call including encoding/head/calibration/hash, result cache disabled. No retrieval/agent/Gemini timing.",
    }
    (safe / (args.variant + "-profile.json")).write_text(json.dumps(summary, indent=2) + "\n")
    (safe / (args.variant + "-profile-records.jsonl")).write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
