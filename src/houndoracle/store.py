"""Local evidence versions, FTS discovery and transactional dependency invalidation."""

import hashlib
import json
import re
import sqlite3
import threading
import time
import uuid

from .models import DocumentInput, Verification


class Store:
    def __init__(self, path=":memory:"):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS documents(
          id TEXT PRIMARY KEY, url TEXT NOT NULL, hash TEXT NOT NULL,
          data TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1);
        CREATE INDEX IF NOT EXISTS doc_url ON documents(url);
        CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(id UNINDEXED, text);
        CREATE TABLE IF NOT EXISTS edges(
          parent TEXT NOT NULL, child TEXT NOT NULL, PRIMARY KEY(parent,child));
        CREATE TABLE IF NOT EXISTS results(
          id TEXT PRIMARY KEY, cache_key TEXT NOT NULL, data TEXT NOT NULL,
          valid INTEGER NOT NULL DEFAULT 1);
        CREATE INDEX IF NOT EXISTS result_key ON results(cache_key);
        CREATE TABLE IF NOT EXISTS dependencies(
          result_id TEXT NOT NULL, document_id TEXT NOT NULL,
          PRIMARY KEY(result_id, document_id));
        CREATE TABLE IF NOT EXISTS jobs(
          id TEXT PRIMARY KEY, status TEXT NOT NULL, request TEXT NOT NULL,
          result TEXT, reason TEXT, created REAL NOT NULL);
        """)

    def close(self):
        self.db.close()

    def _invalidate(self, document_id):
        # Recursive UNION terminates even when an imported graph contains cycles.
        ids = [
            r[0]
            for r in self.db.execute(
                """
        WITH RECURSIVE descendants(id) AS (
          SELECT ? UNION SELECT edges.child FROM edges JOIN descendants ON edges.parent=descendants.id
        ) SELECT id FROM descendants""",
                (document_id,),
            )
        ]
        for doc_id in ids:
            self.db.execute("UPDATE documents SET active=0 WHERE id=?", (doc_id,))
            self.db.execute(
                """UPDATE results SET valid=0 WHERE id IN
              (SELECT result_id FROM dependencies WHERE document_id=?)""",
                (doc_id,),
            )

    def ingest(self, document: DocumentInput, *, now=None):
        now = time.time() if now is None else now
        if document.observed_at > now + 1:
            raise ValueError("Future observation time")
        digest = hashlib.sha256(document.text.encode()).hexdigest()
        doc_id = hashlib.sha256((document.url + "\0" + digest).encode()).hexdigest()
        with self.lock, self.db:
            for parent in document.upstream_ids:
                cyclic = self.db.execute(
                    """
                WITH RECURSIVE ancestors(id) AS (
                  SELECT ? UNION SELECT edges.parent FROM edges JOIN ancestors ON edges.child=ancestors.id
                ) SELECT 1 FROM ancestors WHERE id=? LIMIT 1
                """,
                    (parent, doc_id),
                ).fetchone()
                if cyclic:
                    raise ValueError("Cyclic evidence dependency")
                row = self.db.execute(
                    "SELECT active FROM documents WHERE id=?", (parent,)
                ).fetchone()
                if not row or not row[0] or parent == doc_id:
                    raise ValueError("Upstream must be a distinct active local document")
            previous = self.db.execute(
                "SELECT id FROM documents WHERE url=? AND active=1", (document.url,)
            ).fetchall()
            # Revalidation also invalidates existing verdicts: trust/freshness may change.
            for row in previous:
                self._invalidate(row[0])
            self.db.execute(
                "INSERT OR REPLACE INTO documents VALUES(?,?,?,?,1)",
                (doc_id, document.url, digest, document.model_dump_json()),
            )
            self.db.execute("DELETE FROM passages WHERE id=?", (doc_id,))
            self.db.execute("INSERT INTO passages VALUES(?,?)", (doc_id, document.text))
            self.db.execute("DELETE FROM edges WHERE child=?", (doc_id,))
            for parent in document.upstream_ids:
                self.db.execute("INSERT OR IGNORE INTO edges VALUES(?,?)", (parent, doc_id))
            # New evidence may contradict any prior answer; conservatively invalidate all.
            self.db.execute("UPDATE results SET valid=0")
        return doc_id

    def revoke(self, document_id):
        with self.lock, self.db:
            self._invalidate(document_id)

    def document(self, document_id):
        with self.lock:
            row = self.db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
            return dict(row) if row else None

    def documents(self, *, now=None):
        now = time.time() if now is None else now
        with self.lock:
            rows = self.db.execute("SELECT * FROM documents WHERE active=1 ORDER BY id").fetchall()
            return [dict(r) for r in rows if self._fresh(r["id"], now)]

    def search(self, query, limit=10):
        if not 1 <= limit <= 100:
            raise ValueError("limit must be 1..100")
        tokens = re.findall(r"\w+", query, re.UNICODE)[:40]
        # Function words alone should not make unrelated snapshots look reusable.
        # This changes discovery ranking only, never proposition equivalence.
        stop = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "in",
            "on",
            "at",
            "of",
            "to",
            "for",
            "from",
            "with",
            "by",
            "and",
            "or",
            "not",
            "it",
            "that",
            "this",
        }
        content = [t for t in tokens if t.casefold() not in stop]
        tokens = content or tokens
        if not tokens:
            return []
        expression = " OR ".join('"' + t + '"' for t in tokens)
        with self.lock:
            # Apply the result limit after provenance freshness checks. Otherwise
            # expired high-ranking hits can hide valid evidence further down.
            now = time.time()
            cursor = self.db.execute(
                """SELECT d.*,bm25(passages) AS rank FROM passages
            JOIN documents d ON d.id=passages.id WHERE passages MATCH ? AND active=1
            ORDER BY rank, d.id""",
                (expression,),
            )
            hits = []
            try:
                for row in cursor:
                    if self._fresh(row["id"], now):
                        hits.append({**dict(row), "data": json.loads(row["data"])})
                        if len(hits) == limit:
                            break
            finally:
                cursor.close()
            return hits

    def save(self, key, result):
        result.result_id = uuid.uuid4().hex
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO results VALUES(?,?,?,1)",
                (result.result_id, key, result.model_dump_json()),
            )
            for citation in result.citations:
                self.db.execute(
                    "INSERT OR IGNORE INTO dependencies VALUES(?,?)",
                    (result.result_id, citation.document_id),
                )
        return result

    def cached(self, key, *, now=None):
        now = time.time() if now is None else now
        with self.lock:
            rows = self.db.execute(
                "SELECT data FROM results WHERE cache_key=? AND valid=1 ORDER BY rowid DESC", (key,)
            ).fetchall()
            for row in rows:
                result = Verification.model_validate_json(row[0])
                if result.citations and all(
                    self._fresh(c.document_id, now) for c in result.citations
                ):
                    return result
        return None

    def _fresh(self, document_id, now):
        rows = self.db.execute(
            """
        WITH RECURSIVE ancestors(id) AS (
          SELECT ? UNION SELECT edges.parent FROM edges JOIN ancestors ON edges.child=ancestors.id
        ) SELECT d.* FROM ancestors LEFT JOIN documents d ON d.id=ancestors.id
        """,
            (document_id,),
        ).fetchall()
        return bool(rows) and all(
            r["id"] and r["active"] and json.loads(r["data"])["expires_at"] > now for r in rows
        )

    def explain(self, result_id):
        with self.lock:
            row = self.db.execute("SELECT * FROM results WHERE id=?", (result_id,)).fetchone()
            if not row:
                raise KeyError(result_id)
            result = Verification.model_validate_json(row["data"])
            valid = bool(row["valid"]) and all(
                self._fresh(c.document_id, time.time()) for c in result.citations
            )
            return {"result": result.model_dump(mode="json"), "valid": valid}

    def export(self):
        return {"schema_version": 1, "documents": [json.loads(d["data"]) for d in self.documents()]}

    def import_snapshot(self, snapshot):
        if snapshot.get("schema_version") != 1:
            raise ValueError("Unsupported snapshot version")
        # Preserve provenance but never promote review, authority or sharing permissions.
        pending = {}
        if len(snapshot["documents"]) > 1000:
            raise ValueError("Snapshot exceeds document bound")
        total = 0
        for data in snapshot["documents"]:
            doc = DocumentInput.model_validate(
                {**data, "reviewed": False, "primary": False, "public": False}
            )
            total += len(doc.text.encode())
            if total > 20_000_000 or doc.observed_at > time.time() + 1:
                raise ValueError("Oversized snapshot or future observation")
            digest = hashlib.sha256(doc.text.encode()).hexdigest()
            doc_id = hashlib.sha256((doc.url + "\0" + digest).encode()).hexdigest()
            if doc_id in pending:
                raise ValueError("Duplicate snapshot document identity")
            pending[doc_id] = doc
        if any(not set(doc.upstream_ids).issubset(pending) for doc in pending.values()):
            raise ValueError("Snapshot is missing upstream evidence")
        order, completed = [], set()
        while pending:
            ready = sorted(
                key for key, doc in pending.items() if set(doc.upstream_ids).issubset(completed)
            )
            if not ready:
                raise ValueError("Cyclic snapshot provenance")
            for key in ready:
                order.append(pending.pop(key))
                completed.add(key)
        return [self.ingest(doc) for doc in order]
