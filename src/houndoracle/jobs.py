"""Single-process durable jobs. Restart pauses jobs; paid work is never replayed."""

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from .models import VerifyRequest


class Jobs:
    def __init__(self, oracle):
        self.oracle, self.store = oracle, oracle.store
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oracle")
        self.signals, self.running = {}, set()
        self.lock = threading.RLock()
        with self.store.lock, self.store.db:
            self.store.db.execute(
                "UPDATE jobs SET status='paused',reason='PROCESS_RESTART' WHERE status IN ('running','queued')"
            )

    def submit(self, request: VerifyRequest):
        job_id = uuid.uuid4().hex
        envelope = {
            "request": request.model_dump(mode="json"),
            "used_requests": 0,
            "used_cost": 0.0,
            "used_seconds": 0.0,
        }
        with self.lock, self.store.lock, self.store.db:
            self.store.db.execute(
                "INSERT INTO jobs VALUES(?, 'queued', ?, NULL, NULL, ?)",
                (job_id, json.dumps(envelope), time.time()),
            )
            self._start(job_id)
        return {"id": job_id, "status": "queued"}

    def _start(self, job_id):
        self.signals[job_id] = None
        self.running.add(job_id)
        self.pool.submit(self._run, job_id)

    def get(self, job_id):
        with self.store.lock:
            row = self.store.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise KeyError(job_id)
        return {
            "id": job_id,
            "status": row["status"],
            "reason": row["reason"],
            "result": json.loads(row["result"]) if row["result"] else None,
        }

    def transition(self, job_id, action):
        with self.lock:
            job = self.get(job_id)
            if action in ("pause", "cancel"):
                if job["status"] not in ("queued", "running"):
                    raise ValueError("Only active jobs can pause or cancel")
                self.signals[job_id] = "CANCELLED" if action == "cancel" else "PAUSED"
            elif action == "resume":
                if job["status"] != "paused" or job_id in self.running:
                    raise ValueError("Only settled paused jobs can resume")
                # Restart cannot know whether an in-flight upstream call was billed.
                if job["reason"] in ("PROCESS_RESTART", "RECONCILIATION_REQUIRED"):
                    raise ValueError("Restarted jobs require reconciliation before resubmission")
                with self.store.lock, self.store.db:
                    self.store.db.execute(
                        "UPDATE jobs SET status='queued',reason=NULL WHERE id=?", (job_id,)
                    )
                self._start(job_id)
            else:
                raise ValueError("Unknown job action")
            return self.get(job_id)

    def _run(self, job_id):
        with self.store.lock, self.store.db:
            row = self.store.db.execute("SELECT request FROM jobs WHERE id=?", (job_id,)).fetchone()
            envelope = json.loads(row[0])
            self.store.db.execute("UPDATE jobs SET status='running' WHERE id=?", (job_id,))
        request = dict(envelope["request"])
        request["max_requests"] -= envelope["used_requests"]
        request["max_cost_usd"] = max(0, request["max_cost_usd"] - envelope["used_cost"])
        request["max_seconds"] -= envelope["used_seconds"]
        try:
            if request["max_requests"] < 0 or request["max_seconds"] <= 0:
                raise ValueError("Cumulative budget exhausted")
            question = request.pop("question")
            if request.get("claim"):
                from .models import Claim

                request["claim"] = Claim.model_validate(request["claim"])
            result = self.oracle.verify(
                question, _control=lambda: self.signals.get(job_id), **request
            )
            envelope["used_requests"] += result.usage["requests"]
            envelope["used_cost"] += result.usage["reserved_cost_usd"]
            envelope["used_seconds"] += result.usage["latency_seconds"]
            result.usage.update(
                cumulative_requests=envelope["used_requests"],
                cumulative_reserved_cost_usd=envelope["used_cost"],
                cumulative_seconds=envelope["used_seconds"],
            )
            status = {"CANCELLED": "cancelled", "PAUSED": "paused"}.get(result.reason, "completed")
            with self.store.lock, self.store.db:
                self.store.db.execute(
                    "UPDATE jobs SET status=?,reason=?,request=?,result=? WHERE id=?",
                    (status, result.reason, json.dumps(envelope), result.model_dump_json(), job_id),
                )
        except Exception:  # noqa: BLE001 -- persist uncertain execution for reconciliation
            with self.store.lock, self.store.db:
                self.store.db.execute(
                    "UPDATE jobs SET status='paused',reason='RECONCILIATION_REQUIRED' WHERE id=?",
                    (job_id,),
                )
        finally:
            with self.lock:
                self.running.discard(job_id)

    def close(self):
        with self.lock:
            for job_id in self.running:
                self.signals[job_id] = "PAUSED"
        self.pool.shutdown(wait=True)
