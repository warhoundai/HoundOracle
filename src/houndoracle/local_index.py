"""Build the mandatory local SQLite FTS corpus from operator-supplied snapshots."""

import json
import time
from pathlib import Path
from urllib.parse import urlsplit

from .models import DocumentInput


def import_documents(store, path, *, ttl_seconds=86400):
    path = Path(path)
    if path.stat().st_size > 50_000_000:
        raise ValueError("Import exceeds50MB envelope; split into batches")
    rows = []
    for line in path.open():
        if not line.strip():
            continue
        item = json.loads(line)
        # Content-import is not curator approval or a public contribution.
        now = time.time()
        item.setdefault("origin", urlsplit(item.get("url", "")).hostname or "local-snapshot")
        item.setdefault("observed_at", now)
        item.setdefault("expires_at", now + ttl_seconds)
        item["reviewed"] = False
        item["primary"] = False
        item["public"] = False
        rows.append(DocumentInput.model_validate(item))
        if len(rows) > 1000:
            raise ValueError("Current verification envelope is1000documents per corpus")
    with store.lock:
        existing = store.db.execute("SELECT count(*) FROM documents WHERE active=1").fetchone()[0]
        if existing + len(rows) > 1000:
            raise ValueError("Current corpus limit exceeded; use a separate corpus")
    ids = [store.ingest(row) for row in rows]
    return {
        "indexed": len(ids),
        "document_ids": ids,
        "backend": "sqlite-fts5-bm25",
        "hosted_api_calls": 0,
        "documents_reviewed": False,
    }
