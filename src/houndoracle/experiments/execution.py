"""Bind the official provider adapter to durable per-task reservations and safe continuation."""

import threading
import time
from contextlib import contextmanager
from dataclasses import asdict

from .journal import JournalError, TimeBudgetError
from .vertex import VertexGemini


class JournalBudget:
    def __init__(self, journal, checkpoint=None):
        self.journal = journal
        self.checkpoint = checkpoint
        self.local = threading.local()

    @contextmanager
    def task(self, key, deadline, required_seconds):
        if getattr(self.local, "context", None) is not None:
            raise JournalError("Provider task context cannot be nested")
        self.local.context = {
            "key": key,
            "deadline": deadline,
            "required": required_seconds,
            "call": None,
        }
        try:
            yield
        finally:
            self.local.context = None

    def reserve(self, amount, reason):
        context = getattr(self.local, "context", None)
        if context is None:
            raise JournalError("Provider calls require an active durable task")
        context["call"] = self.journal.reserve(
            context["key"],
            amount,
            remaining_seconds=context["deadline"] - time.monotonic(),
            required_seconds=context["required"],
        )
        if self.checkpoint:
            self.checkpoint.commit(self.journal)


class JournaledGemini:
    def __init__(self, config, journal, *, permission_scope, client=None, checkpoint=None):
        if not isinstance(permission_scope, str) or not permission_scope:
            raise ValueError("An explicit provider permission scope is required")
        journal.bind_provider({"config": asdict(config), "permission_scope": permission_scope})
        self.checkpoint = checkpoint
        self.required_seconds = 165.0 if checkpoint else 125.0
        self.budget = JournalBudget(journal, checkpoint)
        self.journal = journal
        self.delegate = VertexGemini(config, self.budget, client=client)

    def task(self, key, deadline, *, required_seconds=None):
        # countTokens and generation each have a 60-second timeout, plus checkpoint/teardown margin.
        return self.budget.task(
            key, deadline, self.required_seconds if required_seconds is None else required_seconds
        )

    def answer(self, *args, **kwargs):
        context = getattr(self.budget.local, "context", None)
        if context is None:
            raise JournalError("Cannot call provider without a task journal context")
        context["call"] = None
        try:
            result = self.delegate.answer(*args, **kwargs)
        except BaseException:
            if context["call"] is not None:
                self.journal.settle(context["call"], actual_cost=None)
            raise
        self.journal.settle(context["call"], actual_cost=result.get("token_cost_at_list_price_usd"))
        return result


def execute_task(journal, provider, key, inputs, operation, *, deadline):
    journal.add(key, inputs)
    if not journal.claim(key):
        states = {r["key"]: r["state"] for r in journal.tasks()}
        return {"executed": False, "state": states[key], "result": journal.result(key)}
    start = time.perf_counter()
    try:
        with provider.task(key, deadline):
            result = operation()
    except TimeBudgetError:
        journal.defer(key)
        return {"executed": False, "state": "DEFERRED", "result": None}
    except Exception as exc:  # noqa: BLE001 -- preserve per-query failures; never assert a factual NO
        result = {
            "decision": "MAYBE",
            "confidence": None,
            "failure": type(exc).__name__,
            "provider_code": getattr(exc, "code", None),
            "latency_seconds": time.perf_counter() - start,
        }
    journal.finish(key, result, failure=result.get("failure"))
    if provider.checkpoint:
        provider.checkpoint.commit(journal)
    return {
        "executed": True,
        "state": "FAILED" if result.get("failure") else "SUCCEEDED",
        "result": result,
        "wall_seconds_including_checkpoint": time.perf_counter() - start,
    }
