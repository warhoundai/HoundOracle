"""Measure the preserved production baseline on identical evidence, without gold-label access to inference."""

import argparse
import json
import time
from pathlib import Path

import houndoracle
from houndoracle import DocumentInput, Oracle

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for split in ["validation", "calibration"]:
        rows = [
            json.loads(s) for s in (Path(args.data) / (split + ".jsonl")).read_text().splitlines()
        ]
        with (out / ("core-baseline-" + split + ".jsonl")).open("w") as handle:
            for row in rows:
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
                item = {k: row[k] for k in ["id", "group_id", "expected", "category"]}
                item.update(
                    decision=result.decision.value,
                    candidate=result.decision.value,
                    raw_score=None,
                    latency_seconds=time.perf_counter() - start,
                    verification_latency_seconds=result.usage["latency_seconds"],
                    source_urls=list(documents),
                    evidence_group_count=len(documents),
                    cache_hit=result.cache_hit,
                    terminal_reason=result.reason,
                    failure=result.reason
                    if result.reason in {"RETRIEVAL_FAILED", "TIME_EXHAUSTED", "BUDGET_EXHAUSTED"}
                    else None,
                    engine_file=houndoracle.__file__,
                    engine_version=result.versions,
                    baseline_commit="9123fdb",
                    all_in_cost_usd=None,
                    search_api_calls=0,
                    citation_supported=None,
                    input_question=row["question"],
                    retrieval_mode="fixed supplied historical evidence",
                    result=result.model_dump(mode="json"),
                )
                handle.write(json.dumps(item) + "\n")
                oracle.store.close()
        print("Measured baseline", split, len(rows), flush=True)
