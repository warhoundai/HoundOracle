"""A runnable offline example. No model downloads, API keys or network requests."""

import time

from houndoracle import DocumentInput, Oracle

oracle = Oracle()
now = time.time()
oracle.store.ingest(
    DocumentInput(
        url="https://example.org/widget/releases",
        text="Widget was released in 2024.",
        origin="widget-primary-record",
        observed_at=now,
        expires_at=now + 3600,
        primary=True,
        reviewed=True,
    )
)
result = oracle.verify("Was Widget released in 2024?")
print(result.model_dump_json(indent=2))
assert result.reason == "UNCALIBRATED"  # Evidence alone does not establish calibrated correctness.
