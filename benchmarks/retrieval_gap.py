"""Historical supplied-corpus retrieval diagnostic; never a live-web99% qualification."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from houndoracle import Oracle, Store
from houndoracle.experiments.nli import OnnxNLI
from houndoracle.models import DocumentInput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve completed diagnostic")
    source = Path("/workspace/houndoracle-data/campaign-09/validation.jsonl")
    rows = [json.loads(s) for s in source.read_text().splitlines()]
    store = Store()
    oracle = Oracle(store)
    expected_ids = {}
    now = time.time()
    for row in rows:
        ids = []
        for e in row["evidence"]:
            ids.append(
                store.ingest(
                    DocumentInput(
                        url=e["url"],
                        text=e["text"],
                        origin="historical-benchmark-snapshot",
                        observed_at=now,
                        expires_at=now + 3600,
                        primary=False,
                        reviewed=False,
                        retrieved_by="supplied-corpus-diagnostic",
                    )
                )
            )
        expected_ids[row["id"]] = set(ids)
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
    records = []
    for i, row in enumerate(rows):
        start = time.perf_counter()
        hits = oracle.search(row["question"], limit=10)
        lat = time.perf_counter() - start
        gold = expected_ids[row["id"]]
        ranks = [i + 1 for i, h in enumerate(hits) if h["id"] in gold]
        actual = oracle.verify(row["question"])
        # Explicitly separate the research semantic path from the public calibrated engine.
        provided = model.predict(row["question"], row["evidence"])
        passage = [{"page": h["data"]["url"], "text": h["data"]["text"]} for h in hits[:3]]
        retrieved = model.predict(row["question"], passage)
        records.append(
            {k: row[k] for k in ["id", "group_id", "expected"]}
            | {
                "public_engine_decision": actual.decision.value,
                "public_engine_reason": actual.reason,
                "public_engine_correct": actual.decision.value == row["expected"],
                "public_engine_requests": actual.usage["requests"],
                "gold_first_rank": min(ranks) if ranks else None,
                "retrieval_seconds": lat,
                "provided_raw_candidate": provided["candidate"],
                "provided_raw_correct": provided["candidate"] == row["expected"],
                "retrieved_raw_candidate": retrieved["candidate"],
                "retrieved_raw_correct": retrieved["candidate"] == row["expected"],
                "retrieved_context_truncated": retrieved["truncated"],
                "retrieved_documents": len(passage),
            }
        )
        if (i + 1) % 50 == 0:
            print(json.dumps({"scored": i + 1}), flush=True)
    n = len(records)
    definite = [r for r in records if r["expected"] in ["YES", "NO"]]
    report = {
        "rows": n,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "corpus_documents": len(store.documents()),
        "gold_document_recall_on_definitive_labels": {
            str(k): sum(
                r["gold_first_rank"] is not None and r["gold_first_rank"] <= k for r in definite
            )
            / len(definite)
            for k in [1, 3, 10]
        },
        "public_engine_accuracy_all_decisions": sum(r["public_engine_correct"] for r in records)
        / n,
        "public_engine_definitive_coverage": sum(
            r["public_engine_decision"] in ["YES", "NO"] for r in records
        )
        / n,
        "research_encoder_supplied_evidence_raw_accuracy": sum(
            r["provided_raw_correct"] for r in records
        )
        / n,
        "research_encoder_retrieved_top3_raw_accuracy": sum(
            r["retrieved_raw_correct"] for r in records
        )
        / n,
        "retrieved_context_truncation_rate": sum(r["retrieved_context_truncated"] for r in records)
        / n,
        "records": records,
        "qualification99_passed": False,
        "scope": "Reused historical development claims against a supplied corpus containing gold passages plus other queries' evidence. Not web recall, independent held-out evaluation, calibrated confidence, audited citations or production coverage. Raw research classifier is not the public API.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    store.close()


if __name__ == "__main__":
    main()
