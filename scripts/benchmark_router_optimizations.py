"""Local paired evaluator profiling plus clearly marked archived cloud counterfactuals."""

import hashlib
import json
import random
import time
from pathlib import Path

from houndoracle.experiments.head import HeadNLI
from houndoracle.experiments.metrics import paired_comparison, summary
from houndoracle.experiments.nli import OnnxNLI
from houndoracle.experiments.pipeline import ResearchPipeline
from houndoracle.experiments.router import OracleFirstRouter


class FallbackMarker:
    """No provider calls, token costs or provider latency are simulated."""

    def answer(self, question, evidence):
        return {"answer": {"decision": "MAYBE"}, "token_cost_at_list_price_usd": None}


def main():
    started = time.perf_counter()
    source = Path("/workspace/houndoracle-data/campaign-02/validation.jsonl")
    rows = [json.loads(s) for s in source.read_text().splitlines()]
    heads = json.loads(Path("research/experiments/results/trained-aggregators.json").read_text())
    artifact = json.loads(
        Path("research/experiments/results/calibration-logistic.json").read_text()
    )
    pipeline = ResearchPipeline(
        HeadNLI(
            OnnxNLI("/workspace/houndoracle-models/minilm", variant="fp32"),
            heads["logistic"],
            "logistic",
        ),
        artifact,
    )
    routers = {
        "baseline": OracleFirstRouter(pipeline, FallbackMarker(), minimum_confidence=0.85),
        "empty_gate": OracleFirstRouter(
            pipeline, FallbackMarker(), minimum_confidence=0.85, complete_evidence=True
        ),
        "reuse": OracleFirstRouter(
            pipeline,
            FallbackMarker(),
            minimum_confidence=0.85,
            complete_evidence=True,
            evaluation_cache_size=512,
            evaluation_cache_ttl=300,
            permission_scope="frozen-validation-public-only",
        ),
    }
    records = {a: [] for a in routers}
    rng = random.Random(1906)
    for repeat in range(3):
        order = list(rows)
        rng.shuffle(order)
        for row in order:
            arms = list(routers)
            rng.shuffle(arms)
            for arm in arms:
                result = routers[arm].answer(
                    row["question"], row["evidence"], proposition_validated=True
                )
                records[arm].append(
                    {
                        "id": row["id"] + ":" + str(repeat),
                        "original_id": row["id"],
                        "group_id": row["group_id"],
                        "repeat": repeat,
                        "expected": row["expected"],
                        "decision": result["decision"],
                        "latency_seconds": result["latency_seconds"],
                        "cache_hit": result.get("evaluation_cache_hit", False),
                        "empty_gate": result["source"] == "empty_permitted_corpus",
                        "fallback_required": bool(result["model_calls"]),
                        "all_in_cost_usd": None,
                    }
                )
        print(
            json.dumps({"repeat_completed": repeat, "seconds": time.perf_counter() - started}),
            flush=True,
        )
    base = {r["id"]: r for r in records["baseline"]}
    for arm in ["empty_gate", "reuse"]:
        if any(r["decision"] != base[r["id"]]["decision"] for r in records[arm]):
            raise RuntimeError("Verdict drift; reject experiment")
    report = {
        "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "questions": len(rows),
        "repetitions": 3,
        "model_config": pipeline.model.config,
        "calibration_production_qualified": artifact.get("production_qualified"),
        "systems": {
            a: summary(r)
            | {
                "fallbacks_required": sum(x["fallback_required"] for x in r),
                "empty_gate_count": sum(x["empty_gate"] for x in r),
            }
            for a, r in records.items()
        },
        "paired": {
            a: paired_comparison(records["baseline"], records[a], replicates=1000)
            for a in ["empty_gate", "reuse"]
        },
        "reuse_first_pass_including_priming": summary(
            [r for r in records["reuse"] if r["repeat"] == 0]
        ),
        "reuse_later_passes": summary([r for r in records["reuse"] if r["repeat"] > 0]),
        "limitations": [
            "Local evaluator and router only; fallback marker always abstains, no actual Gemini quality or latency measured.",
            "Warm hit rate is constructed; priming first pass included in overall summary.",
            "Historical validation is development evidence, not a new locked evaluation.",
            "No production qualification, all-in cost or central/S3 economics claim.",
        ],
        "seconds": time.perf_counter() - started,
    }
    out = Path("benchmarks/results/campaign-04")
    out.mkdir(parents=True, exist_ok=True)
    (out / "local-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "local-records.jsonl").write_text(
        "".join(json.dumps({"arm": a, **r}) + "\n" for a, rows in records.items() for r in rows)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
