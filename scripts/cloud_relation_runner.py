"""Bounded paired reasoning-level diagnostic; never supplies gold labels to model."""

import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

from houndoracle.experiments.vertex import SpendLedger, VertexConfig, VertexGemini


def emit(kind, **fields):
    print(json.dumps({"campaign": "houndoracle-relation-42", "kind": kind, **fields}), flush=True)


def main():
    payload = json.loads(Path("payload.json").read_text())
    rows = payload["questions"]
    if len(rows) != 24 or len({r["group_id"] for r in rows}) != 24:
        raise ValueError("Require fixed24independent development cases")
    if payload["authorized_reservation_usd"] != 4:
        raise PermissionError("Missing campaign reservation")
    ledger = SpendLedger("/tmp/relation42-ledger.sqlite", authorized_usd=1.04)
    configs = {
        level: VertexConfig(
            project="project-f6e3c8ea-6b96-44a8-afa",
            thinking_level="HIGH",
            evidence_policy=level,
            seed=1991,
            response_format="evidence_indices",
        )
        for level in ["baseline", "relation_audit"]
    }
    tasks = [(level, row) for level in configs for row in rows]
    random.Random(1991).shuffle(tasks)
    emit(
        "configuration",
        models={k: asdict(v) for k, v in configs.items()},
        planned_calls=len(tasks),
        dataset_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
    )
    providers = {level: VertexGemini(config, ledger) for level, config in configs.items()}
    deadline = time.monotonic() + 900
    halted = None
    versions = {}
    for level, row in tasks:
        base = {k: row[k] for k in ["id", "group_id", "expected"]}
        base["thinking_level"] = level
        if halted or time.monotonic() + 125 > deadline:
            emit(
                "query",
                record=base
                | {
                    "decision": "MAYBE",
                    "failure": halted or "DEFERRED_TIME_LIMIT",
                    "request_attempted": False,
                    "latency_seconds": None,
                    "token_cost_at_list_price_usd": 0,
                },
            )
            continue
        config = configs[level]
        emit(
            "reservation",
            id=row["id"],
            thinking_level="HIGH",
            evidence_policy=level,
            maximum_usd=config.maximum_model_cost,
        )
        start = time.perf_counter()
        try:
            provider = providers[level]
            result = provider.answer(row["question"], row["evidence"])
            version = result.get("reported_model_version")
            if version and (
                not version.startswith(config.model)
                or level in versions
                and version != versions[level]
            ):
                halted = "MODEL_VERSION_CHANGED"
            if version:
                versions[level] = version
            record = base | {
                k: result.get(k)
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
            }
            record.update(
                decision=result["answer"]["decision"],
                failure=None if result["output_valid"] else result["failure_reason"],
                request_attempted=True,
                all_in_cost_usd=None,
            )
        except Exception as exc:  # noqa: BLE001 -- preserve failures without credential/query leakage
            # Credential/provider exception strings can contain query or authentication material.
            code = getattr(exc, "code", None)
            code = code if isinstance(code, int) else None
            record = base | {
                "decision": "MAYBE",
                "failure": type(exc).__name__,
                "provider_code": code,
                "request_attempted": True,
                "latency_seconds": time.perf_counter() - start,
                "token_cost_at_list_price_usd": None,
                "maximum_reserved_usd": config.maximum_model_cost,
            }
            if code in [400, 401, 403, 404, 429] or isinstance(exc, (ImportError, PermissionError)):
                halted = "PROVIDER_OR_CONFIGURATION_FAILURE"
        emit("query", record=record)
    emit("completed", planned_slots=len(tasks), halted=halted, versions=versions)
    ledger.close()


if __name__ == "__main__":
    main()
