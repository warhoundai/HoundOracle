"""Proof-producing Node.js release-record tool, not a general truth classifier.

Decisions mean 'according to this dated official release-index snapshot'. LTS
is the index's current marking, not a claim about status on original release day.
No confidence probability or general-engine qualification is inferred.
"""

import hashlib
import json
import re
import sqlite3
import threading
import time
from datetime import date

from ..network import fetch

URL = "https://nodejs.org/dist/index.json"
VERSION = "node-release-records-1"
V = r"v\d{1,3}\.\d{1,3}\.\d{1,3}(?:-[0-9A-Za-z]+(?:\.[0-9A-Za-z]+){0,8})?"
D = r"\d{4}-\d{2}-\d{2}"


def parse(question, depth=0):
    if not isinstance(question, str) or not question.strip() or len(question) > 4000 or depth > 8:
        return None
    text = question.strip().rstrip("?.").strip()
    for prefix, kind in [("Is it false that ", "not"), ("Is it true that ", "identity")]:
        if text.lower().startswith(prefix.lower()):
            child = parse(text[len(prefix) :], depth + 1)
            return (kind, child) if child else None
    # Unparenthesized mixed Boolean scopes are deliberately unsupported.
    if " and " in text and " or " in text:
        return None
    for connector in [" and ", " or "]:
        if connector in text:
            parts = text.split(connector)
            if not 2 <= len(parts) <= 4:
                return None
            children = [parse(p, depth + 1) for p in parts]
            return (connector.strip(), children) if all(children) else None
    match = re.fullmatch(
        rf"(?:Was )?Node\.js ({V}) (not )?released (on|before|after|in) ({D}|\d{{4}}|Node\.js {V})",
        text,
    )
    if match:
        version, negated, op, target = match.groups()
        if target.startswith("Node.js "):
            if op not in {"before", "after"}:
                return None
            value = ("version", target[8:])
        elif len(target) == 4:
            if op != "in":
                return None
            value = ("year", int(target))
        else:
            if op == "in":
                return None
            try:
                value = ("date", date.fromisoformat(target).isoformat())
            except ValueError:
                return None
        atom = ("date", version, op, value)
        return ("not", atom) if negated else atom
    match = re.fullmatch(
        rf"(?:Was )?Node\.js ({V}) released more than (\d{{1,5}}) days before Node\.js ({V})", text
    )
    if match:
        return ("gap", match[1], int(match[2]), match[3])
    match = re.fullmatch(rf"Did Node\.js ({V}) include security fixes", text)
    if match:
        return ("security", match[1])
    match = re.fullmatch(rf"Is Node\.js ({V}) marked LTS in the release index", text)
    if match:
        return ("lts", match[1])
    match = re.fullmatch(
        rf"Does Node\.js ({V}) have a major version greater than (\d{{1,3}})", text
    )
    if match:
        return ("major", match[1], int(match[2]))
    return None


def decode_snapshot(body):
    if not isinstance(body, bytes) or not 1 <= len(body) <= 2_000_000:
        raise ValueError("SNAPSHOT_SIZE")
    text = body.decode("utf-8")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    decoder = json.JSONDecoder(object_pairs_hook=unique_keys)
    i = len(text) - len(text.lstrip())
    if text[i : i + 1] != "[":
        raise ValueError("EXPECTED_ARRAY")
    i += 1
    records = {}
    while True:
        while i < len(text) and text[i].isspace():
            i += 1
        if text[i : i + 1] == "]":
            if text[i + 1 :].strip():
                raise ValueError("TRAILING_CONTENT")
            break
        start = i
        row, i = decoder.raw_decode(text, i)
        if not isinstance(row, dict) or not re.fullmatch(V, str(row.get("version", ""))):
            raise ValueError("INVALID_RELEASE")
        version = row["version"]
        if version in records:
            raise ValueError("DUPLICATE_RELEASE")
        if not isinstance(row.get("date"), str) or not re.fullmatch(D, row["date"]):
            raise ValueError("INVALID_DATE")
        date.fromisoformat(row["date"])
        if "security" in row and type(row["security"]) is not bool:
            raise ValueError("INVALID_SECURITY")
        if (
            "lts" in row
            and row["lts"] is not False
            and not (isinstance(row["lts"], str) and row["lts"].strip())
        ):
            raise ValueError("INVALID_LTS")
        records[version] = {"record": row, "start": start, "end": i, "quote": text[start:i]}
        if len(records) > 5000:
            raise ValueError("RECORD_LIMIT")
        while i < len(text) and text[i].isspace():
            i += 1
        if text[i : i + 1] == ",":
            i += 1
            if text[i:].lstrip().startswith("]"):
                raise ValueError("TRAILING_COMMA")
        elif text[i : i + 1] != "]":
            raise ValueError("INVALID_SEPARATOR")
    if not records:
        raise ValueError("EMPTY_INDEX")
    return records


class ReleaseRecords:
    """SQLite evidence snapshots; cache answers only by re-evaluating exact logic."""

    def __init__(self, path=":memory:", *, transport=fetch, clock=time.time):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.RLock()
        self.transport, self.clock = transport, clock
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS snapshots(hash TEXT PRIMARY KEY,body BLOB,observed REAL,expires REAL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS current(id INTEGER PRIMARY KEY CHECK(id=1),hash TEXT)"
        )
        self.db.commit()

    def close(self):
        self.db.close()

    def refresh(self, *, allow_network=False, ttl_seconds=3600):
        with self.lock:
            return self._refresh(allow_network=allow_network, ttl_seconds=ttl_seconds)

    def _refresh(self, *, allow_network=False, ttl_seconds=3600):
        if not allow_network:
            raise PermissionError("NETWORK_NOT_AUTHORIZED")
        if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= 86400:
            raise ValueError("TTL_LIMIT")
        response = self.transport(
            URL,
            timeout=10,
            max_bytes=2_000_000,
            same_origin_redirects=True,
            accepted_statuses=(200,),
        )
        if response["status"] != 200 or response["url"] != URL:
            raise ValueError("UNEXPECTED_SOURCE")
        body = response["body"]
        decode_snapshot(body)
        digest = hashlib.sha256(body).hexdigest()
        now = self.clock()
        with self.db:
            self.db.execute(
                "INSERT INTO snapshots VALUES(?,?,?,?) ON CONFLICT(hash) DO UPDATE SET observed=excluded.observed,expires=excluded.expires",
                (digest, body, now, now + ttl_seconds),
            )
            self.db.execute(
                "INSERT INTO current VALUES(1,?) ON CONFLICT(id) DO UPDATE SET hash=excluded.hash",
                (digest,),
            )
        return digest

    def verify(self, question, *, allow_network=False):
        with self.lock:
            return self._verify(question, allow_network=allow_network)

    def _verify(self, question, *, allow_network=False):
        started = time.perf_counter()
        query = parse(question)
        base = {
            "question": question,
            "decision": "MAYBE",
            "confidence": None,
            "scope": "Node.js official release-index snapshot only",
            "tool_version": VERSION,
            "production_qualified": False,
            "citations": [],
            "proof": [],
            "requests": 0,
            "cache_hit": False,
        }

        def finish(reason, decision="MAYBE"):
            return base | {
                "reason": reason,
                "decision": decision,
                "latency_seconds": time.perf_counter() - started,
            }

        if query is None:
            return finish("UNSUPPORTED_OR_AMBIGUOUS")
        current = self.db.execute(
            "SELECT s.* FROM snapshots s JOIN current c ON s.hash=c.hash"
        ).fetchone()
        if not current or current[3] <= self.clock():
            if not allow_network:
                return finish("STALE_EVIDENCE" if current else "NO_EVIDENCE")
            base["requests"] = 1
            try:
                self.refresh(allow_network=True)
            except Exception:  # noqa: BLE001 -- credential-safe failure code
                return finish("RETRIEVAL_FAILED")
            current = self.db.execute(
                "SELECT s.* FROM snapshots s JOIN current c ON s.hash=c.hash"
            ).fetchone()
        else:
            base["cache_hit"] = True
        digest, body, observed, expires = current
        if hashlib.sha256(body).hexdigest() != digest:
            return finish("SNAPSHOT_INTEGRITY_FAILED")
        try:
            records = decode_snapshot(body)
        except (ValueError, TypeError, UnicodeError):
            return finish("INVALID_SNAPSHOT")
        base.update(
            snapshot_sha256=digest,
            observed_at=observed,
            expires_at=expires,
            canonical_proposition=query,
        )
        used = set()

        def record(version):
            if version not in records:
                return None
            used.add(version)
            return records[version]["record"]

        def evaluate(q):
            kind = q[0]
            if kind in {"not", "identity"}:
                child = evaluate(q[1])
                return None if child is None else (not child if kind == "not" else child)
            if kind in {"and", "or"}:
                children = [evaluate(c) for c in q[1]]
                if kind == "and":
                    return False if False in children else None if None in children else True
                return True if True in children else None if None in children else False
            r = record(q[1])
            if r is None:
                return None
            if kind == "date":
                left = date.fromisoformat(r["date"])
                op, (target_kind, target) = q[2:]
                if target_kind == "version":
                    other = record(target)
                    if other is None:
                        return None
                    right = date.fromisoformat(other["date"])
                elif target_kind == "year":
                    right = target
                    left = left.year
                else:
                    right = date.fromisoformat(target)
                result = {
                    "on": lambda: left == right,
                    "in": lambda: left == right,
                    "before": lambda: left < right,
                    "after": lambda: left > right,
                }[op]()
                base["proof"].append(
                    {"operation": op, "left": str(left), "right": str(right), "result": result}
                )
                return result
            if kind == "gap":
                other = record(q[3])
                if other is None:
                    return None
                days = (date.fromisoformat(other["date"]) - date.fromisoformat(r["date"])).days
                base["proof"].append(
                    {"operation": "day_difference_gt", "days": days, "threshold": q[2]}
                )
                return days > q[2]
            if kind == "security":
                return r.get("security")
            if kind == "lts":
                return None if "lts" not in r else r["lts"] is not False
            if kind == "major":
                return int(r["version"][1:].split(".")[0]) > q[2]
            return None

        value = evaluate(query)
        if expires <= self.clock():
            return finish("STALE_EVIDENCE")
        base["citations"] = [
            {
                "url": URL,
                "version": v,
                "snapshot_sha256": digest,
                "offset_unit": "unicode_codepoints",
                "start": records[v]["start"],
                "end": records[v]["end"],
                "quote": records[v]["quote"],
            }
            for v in sorted(used)
        ]
        return finish(
            "GROUNDED_RECORD_PROOF" if value is not None else "MISSING_RECORD_OR_FIELD",
            "YES" if value is True else "NO" if value is False else "MAYBE",
        )
