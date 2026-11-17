"""Controlled retrieval regression, not a factual accuracy benchmark."""

import json
import statistics
import time
from pathlib import Path

from houndoracle.models import DocumentInput
from houndoracle.store import Store


def run():
    records = []
    for stale_count in (0, 1, 10, 30):
        for repetition in range(10):
            store = Store()
            try:
                now = time.time()
                for i in range(stale_count):
                    store.ingest(
                        DocumentInput(
                            url=f"https://fixture.example/old/{i}",
                            text="reactor active",
                            origin="fixture.example",
                            observed_at=now - 100,
                            expires_at=now - 1,
                        )
                    )
                expected = store.ingest(
                    DocumentInput(
                        url="https://fixture.example/current",
                        text="reactor active " + "background " * 100,
                        origin="fixture.example",
                        observed_at=now - 100,
                        expires_at=now + 100,
                    )
                )
                start = time.perf_counter()
                rows = store.db.execute("""SELECT d.*,bm25(passages) AS rank
                    FROM passages JOIN documents d ON d.id=passages.id
                    WHERE passages MATCH '"reactor" OR "active"' AND active=1
                    ORDER BY rank LIMIT 1""").fetchall()
                old = [r["id"] for r in rows if json.loads(r["data"])["expires_at"] > time.time()]
                old_time = time.perf_counter() - start
                start = time.perf_counter()
                new = [r["id"] for r in store.search("reactor active", 1)]
                new_time = time.perf_counter() - start
                records.append(
                    {
                        "stale_count": stale_count,
                        "repetition": repetition,
                        "baseline_recalled": expected in old,
                        "candidate_recalled": expected in new,
                        "baseline_seconds": old_time,
                        "candidate_seconds": new_time,
                    }
                )
            finally:
                store.close()
    out = Path("benchmarks/results/campaign-48/search-freshness.json")
    if out.exists():
        raise FileExistsError(out)
    result = {
        "scope": "40 deterministic synthetic retrieval regressions; not world-fact accuracy",
        "baseline_recalled": sum(r["baseline_recalled"] for r in records),
        "candidate_recalled": sum(r["candidate_recalled"] for r in records),
        "cases": len(records),
        "records": records,
        "baseline_median_seconds": statistics.median(r["baseline_seconds"] for r in records),
        "candidate_median_seconds": statistics.median(r["candidate_seconds"] for r in records),
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    run()
