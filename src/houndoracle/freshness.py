"""Bounded conditional source revalidation; revised text loses curator review."""

import json
import time

from .models import DocumentInput
from .network import fetch


def refresh_due(store, budget, *, ttl_seconds=3600, limit=10, transport=fetch):
    if not 1 <= ttl_seconds <= 86400 or not 1 <= limit <= 100:
        raise ValueError("Refresh bounds exceeded")
    with store.lock:
        candidates = store.db.execute(
            "SELECT * FROM documents WHERE active=1 ORDER BY id"
        ).fetchall()
    due = [dict(row) for row in candidates if json.loads(row["data"])["expires_at"] <= time.time()][
        :limit
    ]
    outcomes = []
    for row in due:
        data = json.loads(row["data"])
        headers = {}
        if data.get("etag"):
            headers["If-None-Match"] = data["etag"]
        if data.get("last_modified"):
            headers["If-Modified-Since"] = data["last_modified"]
        budget.reserve(0)
        try:
            response = transport(data["url"], headers=headers, timeout=budget.timeout())
            budget.check_time()
            if response["status"] == 200:
                data["text"] = response["body"].decode("utf-8", errors="strict")
                data["reviewed"] = False
                data["public"] = False
                data["validation_status"] = "unvalidated"
                data["validation_method"] = None
                data["validation_content_hash"] = None
            elif response["status"] != 304:
                raise ValueError("Unexpected revalidation response")
            now = time.time()
            data.update(observed_at=now, expires_at=now + ttl_seconds)
            normalized_headers = {k.lower(): v for k, v in response["headers"].items()}
            data["etag"] = normalized_headers.get("etag", data.get("etag"))
            data["last_modified"] = normalized_headers.get(
                "last-modified", data.get("last_modified")
            )
            doc_id = store.ingest(DocumentInput.model_validate(data))
            outcomes.append(
                {
                    "id": doc_id,
                    "status": "unchanged" if response["status"] == 304 else "review_required",
                }
            )
        except Exception:  # noqa: BLE001 -- failed revalidation must revoke evidence
            # An inaccessible or changed source cannot renew expired evidence.
            store.revoke(row["id"])
            outcomes.append({"id": row["id"], "status": "revalidation_failed"})
    return outcomes
