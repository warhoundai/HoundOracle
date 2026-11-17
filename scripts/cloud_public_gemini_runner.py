"""Bounded matched-evidence Gemini baseline only; no remote Oracle model or hidden retries."""

import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

from houndoracle.experiments.execution import JournaledGemini, execute_task
from houndoracle.experiments.journal import TaskJournal
from houndoracle.experiments.vertex import VertexConfig

PROJECT = "project-f6e3c8ea-6b96-44a8-afa"


def run(payload, *, journal_path, authorized_usd, client=None, emit=None, seconds=900):
    if authorized_usd != 5:
        raise PermissionError(
            "This new pilot requires explicit $5 authorization; old campaigns are closed"
        )
    rows = payload["questions"]
    repetitions = payload["repetitions"]
    if not 1 <= len(rows) <= 48 or type(repetitions) is not int or not 1 <= repetitions <= 3:
        raise ValueError("Pilot workload exceeds48questions/three repetitions")
    if len({r["id"] for r in rows}) != len(rows) or len({r["group_id"] for r in rows}) != len(rows):
        raise ValueError("Repeated/related questions must not masquerade as independent cases")
    config = VertexConfig(project=PROJECT, seed=1930, response_format="evidence_indices")
    tasks = [(repeat, row) for repeat in range(repetitions) for row in rows]
    random.Random(1930).shuffle(tasks)
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    journal = TaskJournal(
        journal_path,
        configuration={"dataset": digest, "model": asdict(config), "repetitions": repetitions},
        maximum_calls=144,
        authorized_usd=3.10,
    )
    outputs = []
    emit = emit or (
        lambda kind, **fields: print(
            json.dumps({"campaign": "houndoracle-public-14", "kind": kind, **fields}), flush=True
        )
    )
    provider = JournaledGemini(
        config, journal, permission_scope="public-vitaminc-conditional-campaign14", client=client
    )
    reserve = provider.budget.reserve

    def logged_reserve(amount, reason):
        reserve(amount, reason)
        context = provider.budget.local.context
        emit("reservation", task=context["key"], journal=journal.summary())

    provider.budget.reserve = logged_reserve
    for repeat, row in tasks:
        journal.add(
            f"{row['id']}:{repeat}", {"question": row["question"], "evidence": row["evidence"]}
        )
    emit(
        "configuration",
        model=asdict(config),
        maximum_calls=144,
        maximum_model_reservation_usd=3.096576,
        dataset_sha256=digest,
        planned_tasks=len(tasks),
    )
    deadline = time.monotonic() + min(seconds, 900)
    version = None
    try:
        for repeat, row in tasks:
            key = f"{row['id']}:{repeat}"

            def operation(row=row):
                # Gold expected labels and local Oracle predictions are never supplied to Gemini.
                result = provider.answer(row["question"], row["evidence"])
                usage = result["raw_response"].get("usage_metadata", {})
                return {
                    "decision": result["answer"]["decision"],
                    "answer": result["answer"],
                    "failure": None if result["output_valid"] else result["failure_reason"],
                    **{
                        k: result[k]
                        for k in [
                            "latency_seconds",
                            "input_tokens",
                            "output_tokens",
                            "reasoning_tokens",
                            "reported_model_version",
                            "token_cost_at_list_price_usd",
                            "reserved_usd",
                            "output_valid",
                        ]
                    },
                    "cached_input_tokens": usage.get("cached_content_token_count"),
                    "all_in_cost_usd": None,
                }

            result = execute_task(
                journal,
                provider,
                key,
                {"question": row["question"], "evidence": row["evidence"]},
                operation,
                deadline=deadline,
            )
            record = {
                "id": row["id"],
                "group_id": row["group_id"],
                "expected": row["expected"],
                "repeat": repeat,
                **result,
            }
            outputs.append(record)
            emit("query", record=record, journal=journal.summary())
            output = result.get("result") or {}
            reported = output.get("reported_model_version")
            if reported:
                if (
                    not reported.startswith("gemini-3.8-flash")
                    or version is not None
                    and reported != version
                ):
                    emit("halt", reason="MODEL_VERSION_CHANGED", reported=reported)
                    break
                if version is None:
                    version = reported
                    emit("observed_version_pin", version=version)
            if output.get("failure") in ["ModuleNotFoundError", "ImportError"]:
                emit("halt", reason="PACKAGING_FAILURE", failure=output["failure"])
                break
            if output.get("provider_code") in [400, 401, 403, 404, 429]:
                emit(
                    "halt",
                    reason="PROVIDER_CONFIGURATION_OR_QUOTA_FAILURE",
                    code=output["provider_code"],
                )
                break
            if result["state"] == "DEFERRED":
                emit("halt", reason="INSUFFICIENT_TIME_FOR_SAFE_NEW_REQUEST")
                break
        summary = journal.summary()
        emit("summary", journal=summary, task_states=journal.tasks(), version=version)
        return {"records": outputs, "summary": summary}
    finally:
        journal.close()


if __name__ == "__main__":
    payload = json.loads(Path(sys.argv[1]).read_text())
    run(
        payload,
        journal_path="/tmp/houndoracle-public-14.sqlite",
        authorized_usd=float(os.environ.get("HOUNDORACLE_PILOT_APPROVED_USD", "0")),
    )
