"""One-shot frozen heldout evaluation. Refuses reruns; never selects/tunes on results."""

import hashlib
import json
import time
from pathlib import Path

from houndoracle import DocumentInput, Oracle
from houndoracle.experiments.dataset import digest
from houndoracle.experiments.head import infer_head
from houndoracle.experiments.metrics import cluster_interval, summary
from houndoracle.experiments.nli import OnnxNLI
from houndoracle.experiments.pipeline import apply

DATA = Path("/workspace/houndoracle-data/campaign-02")
OUT = Path("/workspace/houndoracle-data/locked-results")
REPORT = Path("benchmarks/results/campaign-02")
VARIANTS = ["fp32", "int8", "logistic", "mlp", "logistic-int8", "mlp-int8"]


def main():
    if OUT.exists():
        raise RuntimeError("Locked evaluation already started; do not tune and replay it")
    artifacts = {
        v: json.loads(Path(f"research/experiments/results/calibration-{v}.json").read_text())
        for v in VARIANTS
    }
    heads = json.loads(Path("research/experiments/results/trained-aggregators.json").read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())
    freeze = {
        "baseline_commit": "9123fdb",
        "dataset_evaluation_digest": manifest["splits"]["evaluation"]["digest"],
        "variants": {v: digest(a) for v, a in artifacts.items()},
        "head_digest": digest(heads),
        "selection": "All four calibrated head configurations retained as a validation Pareto set; raw encoders are negative controls.",
        "production_replacement": False,
        "locked_test_tuning_permitted": False,
    }
    Path("research/experiments/final-freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    OUT.mkdir()
    REPORT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(s) for s in (DATA / "evaluation.jsonl").read_text().splitlines()]
    if digest(rows) != freeze["dataset_evaluation_digest"]:
        raise ValueError("Locked dataset digest mismatch")
    collected = {v: [] for v in VARIANTS}
    collected["core-baseline"] = []
    models = {
        v: OnnxNLI("/workspace/houndoracle-models/minilm", variant=v) for v in ["fp32", "int8"]
    }
    for i, row in enumerate(rows):
        common = {k: row[k] for k in ["id", "group_id", "expected", "category"]}
        common.update(
            input_question=row["question"],
            source_urls=[e["url"] for e in row["evidence"]],
            evidence_group_count=len({e["page"] for e in row["evidence"]}),
            cache_hit=False,
            all_in_cost_usd=None,
            compute_cost_usd=None,
            storage_cost_usd=None,
            search_api_calls=0,
            model_output_tokens=0,
            reasoning_tokens=0,
            model_cost_usd=0,
            search_cost_usd=0,
            retrieval_recall=None,
            retrieval_mode="supplied historical evidence; no live retrieval",
        )
        # Stable core: unchanged decision policy, uncalibrated, original raw question.
        oracle = Oracle()
        now = time.time()
        documents = {}
        for e in row["evidence"]:
            if e["text"].strip():
                documents.setdefault(e["url"], []).append(e["text"])
        start = time.perf_counter()
        for url, texts in documents.items():
            oracle.store.ingest(
                DocumentInput(
                    url=url,
                    text="\n".join(texts),
                    origin=url,
                    observed_at=now,
                    expires_at=now + 3600,
                    reviewed=True,
                    primary=False,
                )
            )
        result = oracle.verify(row["question"])
        oracle.store.close()
        collected["core-baseline"].append(
            {
                **common,
                "decision": result.decision.value,
                "latency_seconds": time.perf_counter() - start,
                "terminal_reason": result.reason,
                "failure": None,
                "candidate": result.decision.value,
                "confidence": result.confidence,
            }
        )
        for variant, model in models.items():
            start = time.perf_counter()
            base = model.predict(row["question"], row["evidence"])
            base_seconds = time.perf_counter() - start
            names = [
                variant,
                "logistic" if variant == "fp32" else "logistic-int8",
                "mlp" if variant == "fp32" else "mlp-int8",
            ]
            for name in names:
                started = time.perf_counter()
                prediction = (
                    base
                    if name == variant
                    else infer_head(
                        base,
                        heads[name.split("-")[0]],
                        evidence_group_count=common["evidence_group_count"],
                    )
                )
                prediction = apply(prediction, artifacts[name])
                item = {
                    **common,
                    **prediction,
                    "latency_seconds": base_seconds + time.perf_counter() - started,
                    "model_input_tokens": base.get("token_count"),
                    "failure": None,
                    "citation_supported": None,
                    "label_agreement_on_supplied_evidence": prediction["decision"] in ("YES", "NO")
                    and prediction["decision"] == row["expected"],
                    "citation_audit_method": "original complete gold evidence set proxy; not a second human citation audit",
                    "citations": [
                        {"url": e["url"], "quote": e["text"], "sentence_id": e["sentence_id"]}
                        for e in row["evidence"]
                    ],
                    "engine_config": artifacts[name]["model_config"],
                }
                collected[name].append(item)
        if (i + 1) % 100 == 0:
            print("Locked evaluation", i + 1, "/", len(rows), flush=True)
    report = {
        "freeze": freeze,
        "systems": {},
        "gemini": "not executed: project CPU quota blocked both job submissions",
        "scope": "historical FEVER evidence-conditioned task; not live-world research or complete AI-agent workflow",
        "unknowns": [
            "all-in monetary costs",
            "S3 reuse economics",
            "Gemini relative accuracy/cost/latency",
            "independent human citation re-audit",
        ],
    }
    for name, values in collected.items():
        path = OUT / (name + ".jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in values))
        report["systems"][name] = {
            **summary(values),
            "intervals": {
                m: cluster_interval(values, m)
                for m in [
                    "accuracy_all_decisions",
                    "definitive_coverage",
                    "selective_risk",
                    "correct_definitive_fraction",
                ]
            },
            "raw_records": str(path),
            "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        # Commit IDs/outcomes/timings only. Full question/evidence records remain under external data rights.
        redacted = [
            {k: v for k, v in r.items() if k not in ["input_question", "citations"]} for r in values
        ]
        (REPORT / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in redacted))
    (REPORT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {
                    k: v
                    for k, v in value.items()
                    if k
                    in [
                        "accuracy_all_decisions",
                        "definitive_coverage",
                        "selective_risk",
                        "correct_definitive_fraction",
                        "latency_seconds",
                    ]
                }
                for name, value in report["systems"].items()
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
