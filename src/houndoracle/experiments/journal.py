"""Private durable benchmark state; uncertainty consumes reservations and is never silently retried."""

import hashlib
import json
import math
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class JournalError(RuntimeError):
    pass


class TimeBudgetError(JournalError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class TaskJournal:
    def __init__(self, path, *, configuration, maximum_calls, authorized_usd, worker_id=None):
        if (
            type(maximum_calls) is not int
            or maximum_calls < 0
            or not math.isfinite(authorized_usd)
            or authorized_usd < 0
        ):
            raise ValueError("Invalid campaign resource envelope")
        path = str(path)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                os.close(fd)
            except FileExistsError:
                if os.stat(path).st_mode & 0o077:
                    raise PermissionError(
                        "Existing journal must have private filesystem permissions"
                    )
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.worker = worker_id or uuid.uuid4().hex
        self.db.execute("PRAGMA synchronous=FULL")
        expected = canonical(
            {
                "configuration": configuration,
                "maximum_calls": maximum_calls,
                "authorized_usd": authorized_usd,
            }
        )
        try:
            with self.transaction():
                tables = {
                    r[0]
                    for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")
                }
                if tables and "oracle_campaign_meta" not in tables:
                    raise ValueError("Refuse to modify a different database")
                self.db.execute(
                    "CREATE TABLE IF NOT EXISTS oracle_campaign_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)"
                )
                self.db.execute(
                    "CREATE TABLE IF NOT EXISTS oracle_tasks(key TEXT PRIMARY KEY,input_hash TEXT NOT NULL,state TEXT NOT NULL,worker TEXT,started REAL,finished REAL,result TEXT,failure TEXT)"
                )
                self.db.execute(
                    "CREATE TABLE IF NOT EXISTS oracle_calls(id INTEGER PRIMARY KEY,task TEXT NOT NULL,reserved REAL NOT NULL,actual REAL,state TEXT NOT NULL,started REAL NOT NULL,finished REAL)"
                )
                old = self.db.execute(
                    "SELECT value FROM oracle_campaign_meta WHERE key='configuration'"
                ).fetchone()
                if old and old[0] != expected:
                    raise ValueError(
                        "Campaign inputs, policy or authorization changed; use a separate journal"
                    )
                self.db.execute(
                    "INSERT OR IGNORE INTO oracle_campaign_meta VALUES('configuration',?)",
                    (expected,),
                )
            self.db.execute("PRAGMA journal_mode=WAL")
        except BaseException:
            self.db.close()
            raise
        self.maximum_calls, self.cap = maximum_calls, authorized_usd

    @contextmanager
    def transaction(self):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise

    def bind_provider(self, configuration):
        expected = canonical(configuration)
        with self.transaction():
            old = self.db.execute(
                "SELECT value FROM oracle_campaign_meta WHERE key='provider'"
            ).fetchone()
            if old and old[0] != expected:
                raise ValueError(
                    "Provider settings or permission scope changed; use a separate campaign"
                )
            self.db.execute(
                "INSERT OR IGNORE INTO oracle_campaign_meta VALUES('provider',?)", (expected,)
            )

    def add(self, key, inputs):
        digest = hashlib.sha256(canonical(inputs).encode()).hexdigest()
        with self.transaction():
            row = self.db.execute(
                "SELECT input_hash FROM oracle_tasks WHERE key=?", (key,)
            ).fetchone()
            if row and row[0] != digest:
                raise ValueError("Same task key has different proposition, evidence or settings")
            self.db.execute(
                "INSERT OR IGNORE INTO oracle_tasks(key,input_hash,state) VALUES(?,?,'PENDING')",
                (key, digest),
            )

    def _owned(self, key):
        row = self.db.execute("SELECT * FROM oracle_tasks WHERE key=?", (key,)).fetchone()
        if row is None or row["state"] != "RUNNING" or row["worker"] != self.worker:
            raise JournalError("Task is not owned by this active worker")
        return row

    def claim(self, key):
        with self.transaction():
            halted = self.db.execute(
                "SELECT value FROM oracle_campaign_meta WHERE key='halted'"
            ).fetchone()
            if halted:
                raise JournalError("Campaign halted after a cost-envelope violation")
            changed = self.db.execute(
                "UPDATE oracle_tasks SET state='RUNNING',worker=?,started=? WHERE key=? AND state='PENDING'",
                (self.worker, time.time(), key),
            ).rowcount
        return bool(changed)

    def reserve(self, key, amount, *, remaining_seconds, required_seconds):
        if (
            not all(math.isfinite(v) for v in [amount, remaining_seconds, required_seconds])
            or amount <= 0
            or required_seconds < 0
        ):
            raise ValueError("Invalid request reservation or time limit")
        if remaining_seconds < required_seconds:
            raise TimeBudgetError(
                "Do not launch a request that cannot fit the remaining worker window"
            )
        with self.transaction():
            self._owned(key)
            if self.db.execute("SELECT 1 FROM oracle_campaign_meta WHERE key='halted'").fetchone():
                raise JournalError("Campaign halted")
            if self.db.execute(
                "SELECT 1 FROM oracle_calls WHERE task=? AND state='RESERVED'", (key,)
            ).fetchone():
                raise JournalError("Task already has an outstanding provider call")
            count, used = self.db.execute(
                "SELECT count(*),coalesce(sum(reserved),0) FROM oracle_calls"
            ).fetchone()
            if count >= self.maximum_calls or used + amount > self.cap + 1e-12:
                raise PermissionError("Durable campaign spending/request cap exhausted")
            call = self.db.execute(
                "INSERT INTO oracle_calls(task,reserved,state,started) VALUES(?,?,'RESERVED',?)",
                (key, amount, time.time()),
            )
            return call.lastrowid

    def settle(self, call_id, *, actual_cost=None):
        if actual_cost is not None and (not math.isfinite(actual_cost) or actual_cost < 0):
            raise ValueError("Invalid observed cost")
        exceeded = False
        with self.transaction():
            row = self.db.execute("SELECT * FROM oracle_calls WHERE id=?", (call_id,)).fetchone()
            if row is None or row["state"] != "RESERVED":
                raise JournalError("Call already settled or uncertain; cannot overwrite it")
            self._owned(row["task"])
            self.db.execute(
                "UPDATE oracle_calls SET actual=?,state=?,finished=? WHERE id=?",
                (
                    actual_cost,
                    "SETTLED" if actual_cost is not None else "UNCERTAIN",
                    time.time(),
                    call_id,
                ),
            )
            if actual_cost is not None and actual_cost > row["reserved"] + 1e-12:
                self.db.execute(
                    "INSERT OR REPLACE INTO oracle_campaign_meta VALUES('halted','COST_ENVELOPE_VIOLATED')"
                )
                exceeded = True
        if exceeded:
            raise JournalError("Observed cost exceeded reservation; preserved and halted campaign")

    def finish(self, key, result, *, failure=None):
        encoded = canonical(result)
        with self.transaction():
            self._owned(key)
            if self.db.execute(
                "SELECT 1 FROM oracle_calls WHERE task=? AND state='RESERVED'", (key,)
            ).fetchone():
                raise JournalError("Settle or mark provider uncertainty before finishing the task")
            self.db.execute(
                "UPDATE oracle_tasks SET state=?,result=?,failure=?,finished=? WHERE key=?",
                ("FAILED" if failure else "SUCCEEDED", encoded, failure, time.time(), key),
            )

    def defer(self, key):
        with self.transaction():
            self._owned(key)
            if self.db.execute("SELECT 1 FROM oracle_calls WHERE task=?", (key,)).fetchone():
                raise JournalError("A task with provider work cannot be reset to pending")
            self.db.execute(
                "UPDATE oracle_tasks SET state='PENDING',worker=NULL,started=NULL WHERE key=?",
                (key,),
            )

    def recover_worker(self, worker_id, *, confirmed_stopped=False):
        if not confirmed_stopped:
            raise PermissionError(
                "Recovery requires a confirmed stopped worker, not an assumed timeout"
            )
        with self.transaction():
            self.db.execute(
                "UPDATE oracle_calls SET state='UNCERTAIN',finished=? WHERE state='RESERVED' AND task IN (SELECT key FROM oracle_tasks WHERE worker=? AND state='RUNNING')",
                (time.time(), worker_id),
            )
            return self.db.execute(
                "UPDATE oracle_tasks SET state='UNCERTAIN',failure='PROCESS_INTERRUPTED',finished=? WHERE worker=? AND state='RUNNING'",
                (time.time(), worker_id),
            ).rowcount

    def tasks(self):
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT * FROM oracle_tasks ORDER BY key")]

    def result(self, key):
        with self.lock:
            row = self.db.execute(
                "SELECT state,result FROM oracle_tasks WHERE key=?", (key,)
            ).fetchone()
            return (
                json.loads(row["result"])
                if row and row["state"] in ["SUCCEEDED", "FAILED"]
                else None
            )

    def summary(self):
        with self.lock:
            rows = self.db.execute("SELECT reserved,actual,state FROM oracle_calls").fetchall()
            counts = dict(
                self.db.execute("SELECT state,count(*) FROM oracle_tasks GROUP BY state").fetchall()
            )
        known = sum(r["actual"] for r in rows if r["actual"] is not None)
        unknown = sum(r["reserved"] for r in rows if r["actual"] is None)
        return {
            "tasks": counts,
            "reserved_calls": len(rows),
            "reserved_usd": sum(r["reserved"] for r in rows),
            "reported_token_cost_usd": known,
            "unknown_usage_calls": sum(r["actual"] is None for r in rows),
            "token_cost_bounds_usd": [known, known + unknown],
            "actual_invoiced_all_in_cost_usd": None,
        }

    def snapshot(self, path):
        with self.lock:
            target = sqlite3.connect(str(path))
            try:
                target.execute("PRAGMA synchronous=FULL")
                self.db.backup(target)
            finally:
                target.close()

    def close(self):
        self.db.close()
