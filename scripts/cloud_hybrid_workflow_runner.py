"""Actual local-encoder plus optional Vertex fallback in one bounded cloud worker."""

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

from houndoracle.experiments.execution import JournaledGemini, execute_task
from houndoracle.experiments.head import HeadNLI
from houndoracle.experiments.journal import TaskJournal
from houndoracle.experiments.nli import OnnxNLI
from houndoracle.experiments.pipeline import ResearchPipeline
from houndoracle.experiments.vertex import VertexConfig


def run(
    payload,
    *,
    authorized_usd,
    model_directory="/tmp/model",
    model=None,
    client=None,
    emit=None,
    seconds=1500,
    journal_path="/tmp/workflow18.sqlite",
):
    if authorized_usd != 8:
        raise PermissionError("Requires reserved eight-dollar campaign tranche")
    questions = payload["questions"]
    if (
        len(questions) > 48
        or payload["repetitions"] != 3
        or len({r["group_id"] for r in questions}) != len(questions)
    ):
        raise ValueError("Invalid workflow cohort")
    emit = emit or (
        lambda kind, **fields: print(
            json.dumps({"campaign": "houndoracle-workflow-18", "kind": kind, **fields}), flush=True
        )
    )
    initialization = time.perf_counter()
    if model is None:
        model = ResearchPipeline(
            HeadNLI(
                OnnxNLI(
                    model_directory,
                    variant="fp32",
                    threads=1,
                    max_length=256,
                    reviewed_manifest=payload["manifest"],
                ),
                payload["head"],
                "continued-mixture",
            ),
            payload["calibration"],
        )
    # Verify exact frozen decisions before paid calls; this also warms the encoder, not result cache.
    for row in questions:
        actual = model.verify_evidence(
            row["question"],
            row["evidence"],
            population=payload["calibration"]["population"],
            use_cache=False,
        )
        if actual["decision"] != payload["frozen_local_decisions"][row["id"]]:
            raise ValueError("Cloud runtime differs from frozen local decision policy")
    emit(
        "initialization",
        seconds=time.perf_counter() - initialization,
        model_config=model.model.config,
        scope="Warm model, empty result cache at measured start; initialization/preflight reported separately",
    )
    config = VertexConfig(
        project="project-f6e3c8ea-6b96-44a8-afa", seed=1930, response_format="evidence_indices"
    )
    journal = TaskJournal(
        journal_path,
        configuration={
            "dataset": payload["dataset_sha256"],
            "model": asdict(config),
            "oracle": model.model.config,
            "repetitions": 3,
        },
        maximum_calls=288,
        authorized_usd=6.20,
    )
    provider = JournaledGemini(
        config, journal, permission_scope="public-vitaminc-actual-hybrid-18", client=client
    )
    reserve = provider.budget.reserve

    def logged_reserve(amount, reason):
        reserve(amount, reason)
        emit("reservation", task=provider.budget.local.context["key"], journal=journal.summary())

    provider.budget.reserve = logged_reserve
    rng = random.Random(1934)
    tasks = []
    for repeat in range(3):
        rows = questions.copy()
        rng.shuffle(rows)
        for row in rows:
            arms = ["gemini", "hybrid"]
            rng.shuffle(arms)
            tasks.extend((repeat, row, arm) for arm in arms)
    for repeat, row, arm in tasks:
        journal.add(
            f"{row['id']}:{repeat}:{arm}",
            {"question": row["question"], "evidence": row["evidence"], "arm": arm},
        )
    emit(
        "configuration",
        planned_tasks=len(tasks),
        model=asdict(config),
        maximum_model_calls=288,
        dataset_sha256=payload["dataset_sha256"],
    )
    deadline = time.monotonic() + min(seconds, 1500)

    def remote(row):
        result = provider.answer(row["question"], row["evidence"])
        return {
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
        } | {
            "decision": result["answer"]["decision"],
            "answer": result["answer"],
            "failure": None if result["output_valid"] else result["failure_reason"],
            "source": "gemini",
            "model_calls": 1,
            "all_in_cost_usd": None,
        }

    outputs = []
    try:
        for repeat, row, arm in tasks:
            if deadline - time.monotonic() < 125:
                emit("halt", reason="TIME_RESERVE_EXHAUSTED")
                break
            key = f"{row['id']}:{repeat}:{arm}"

            def operation(row=row, arm=arm):
                local = None
                if arm == "hybrid":
                    local = model.verify_evidence(
                        row["question"],
                        row["evidence"],
                        population=payload["calibration"]["population"],
                        use_cache=True,
                    )
                    if local["decision"] in ["YES", "NO"]:
                        return {
                            "decision": local["decision"],
                            "source": "oracle",
                            "local": local,
                            "model_calls": 0,
                            "token_cost_at_list_price_usd": 0,
                            "all_in_cost_usd": None,
                            "failure": None,
                        }
                return remote(row) | {"local": local}

            started = time.perf_counter()
            envelope = execute_task(
                journal,
                provider,
                key,
                {"question": row["question"], "evidence": row["evidence"], "arm": arm},
                operation,
                deadline=deadline,
            )
            record = (
                {k: row[k] for k in ["id", "group_id", "expected"]}
                | {
                    "repeat": repeat,
                    "arm": arm,
                    "workflow_latency_seconds": time.perf_counter() - started,
                }
                | envelope
            )
            outputs.append(record)
            emit("query", record=record, journal=journal.summary())
            output = envelope.get("result") or {}
            if (
                output.get("reported_model_version") not in [None, "gemini-3.8-flash"]
                or output.get("provider_code") in [400, 401, 403, 404, 429]
                or output.get("failure") in ["ModuleNotFoundError", "ImportError"]
            ):
                emit("halt", reason="MODEL_OR_CONFIGURATION_FAILURE")
                break
        emit("summary", journal=journal.summary(), tasks=journal.tasks())
        return {"records": outputs, "summary": journal.summary()}
    finally:
        journal.close()


if __name__ == "__main__":
    run(json.loads(Path("payload.json").read_text()), authorized_usd=8)
