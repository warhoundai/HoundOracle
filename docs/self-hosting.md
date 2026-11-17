# Offline self-hosted Oracle

For the intended local-with-internet deployment, use [self-hosted web search](web-search.md). This guide covers the disconnected profile.

The required serving path runs on your own machine: SQLite FTS5/BM25 search, evidence/provenance storage, deterministic verification and local aggregation. It requires no Exa, Google, OpenAI or central Oracle account. `Oracle()` now defaults to `local_only=True`; hosted providers cannot be attached in that mode. Commands without `--web` enable this offline policy. Environment API keys do not activate a cloud backend.

This is an index of documents you import/crawl, not a locally hosted Google index. Use your own HoundCrawl/HoundScrapper process or other permitted source downloader to prepare snapshots. Crawl/update separately; serving and inference can then run without internet. Current verification corpus envelope:1,000 documents/5MB. This is not yet a whole-web-scale index or a claim that arbitrary natural-language facts are fully supported.

## Build and run

```bash
# Generate your own local service access token (not a vendor API key).
export HOUNDORACLE_API_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
docker compose -f deployment/selfhost/compose.yaml build
# snapshots.jsonl contains one {"url":...,"text":...} object per line.
docker compose -f deployment/selfhost/compose.yaml run --rm \
  -v "$PWD/snapshots.jsonl:/input/snapshots.jsonl:ro" oracle index /input/snapshots.jsonl
docker compose -f deployment/selfhost/compose.yaml up -d
```

The internal Docker network blocks outbound routing, and the published port binds only to127.0.0.1. Evidence lives in the persistent volume. The container runs as UID10001 with read-only root, no Linux capabilities and writable temporary/data mounts. Installation/build may download Python dependencies; runtime does not need hosted APIs. On networks that intercept TLS, install your organization's trusted CA for builds; do not disable certificate verification.

```bash
curl http://127.0.0.1:8083/v1/search \
  -H "Authorization: Bearer $HOUNDORACLE_API_KEY" \
  -H 'Content-Type: application/json' -d '{"query":"Python release"}'
```

SDK equivalent:

```python
from houndoracle import Oracle, Store
from houndoracle.local_index import import_documents
oracle = Oracle(Store("oracle.sqlite"))
import_documents(oracle.store, "snapshots.jsonl")
results = oracle.search("Python release")
answer = oracle.verify("Was Python 3.12.0 released in 2023?", min_confidence=0.99)
```

Import does not mark scraped text reviewed, primary, public or true. Curator review and independently qualifying calibration are still required for definitive decisions. The existing `ingest` command is the trusted local-curator path. A confidence threshold of.99 does not itself establish99% accuracy. The default engine abstains without qualifying calibration; keep that behavior.

## Accuracy acceptance target

The new precision gate tests a frozen policy with independently labeled, source/event-disjoint questions, independently reviewed supporting spans and a predeclared coverage floor. At99% precision and one-sided95% confidence, even zero errors needs at least299 independent accepted questions. Correlated paraphrases cannot inflate that count. Reject models that achieve a misleading perfect score by answering nothing. Finite tests do not prove100% universal future correctness.

The current600-query neural snapshot regression has not met this target. Its declared calibration population is historical supplied evidence, not arbitrary current web facts. Do not advertise99–100% or a public performance release until independent gates pass.

## Earlier hosted experiments

Exa/HoundSearch/Wikipedia adapters remain isolated research integrations for historical SDK experiments. The supported `--web` switch uses self-hosted SearXNG; it does not activate Exa or Gemini. They are not required or enabled by the self-hosted serving profile, and no supplied vendor credentials are included in the image or EC2 test bundle. A Google-backed SearXNG server would still query external engines, so it is not the fully local index required here.
