# ruff: noqa: F821
# Template uses only the original submitted helpers/modules, not newer source policy.
"""Bounded sequential continuation; errors are records, never evidence of falsehood."""


def main():
    ledger = SpendLedger("/tmp/recovery-spend.sqlite", authorized_usd=6.193152)
    ledger.reserve(179 * 0.021504, "178 completed + one uncertain prior generation reservation")
    gemini = FreshCounter(VertexGemini(VertexConfig(project=PAYLOAD["project"], seed=1907), ledger))
    gemini.attempts = 179
    oracles = prepare_oracles()
    router = OracleFirstRouter(
        oracles["logistic"],
        gemini,
        minimum_confidence=0.85,
        complete_evidence=True,
        evaluation_cache_size=0,
    )
    lookup = {r["id"]: r for r in PAYLOAD["confirmation"]}
    emit(
        "recovery_started", prior_attempts=179, unattempted_tasks=len(RECOVERY), timeout_seconds=400
    )
    for item in RECOVERY:
        row = lookup[item["id"]]
        start = time.perf_counter()
        before = gemini.attempts
        try:
            if item["arm"] == "baseline" and row["evidence"] == []:
                result = {
                    "decision": "MAYBE",
                    "gemini": None,
                    "oracle": None,
                    "source": "empty_permitted_corpus",
                    "latency_seconds": time.perf_counter() - start,
                    "token_cost_usd": 0,
                    "model_calls": 0,
                    "failure": None,
                }
            else:
                result = (
                    standalone(gemini, row)
                    if item["arm"] == "baseline"
                    else router.answer(row["question"], row["evidence"], proposition_validated=True)
                )
        except Exception as exc:  # noqa: BLE001 -- preserve each failure without treating it as evidence
            result = {
                "decision": "MAYBE",
                "gemini": None,
                "oracle": None,
                "source": "provider_or_evaluator_failure",
                "latency_seconds": time.perf_counter() - start,
                "token_cost_usd": None,
                "model_calls": gemini.attempts - before,
                "failure": type(exc).__name__,
                "provider_code": getattr(exc, "code", None),
                "reserved_usd": (gemini.attempts - before) * 0.021504,
            }
        emit(
            "query",
            record={
                **result,
                **item,
                "group_id": row["group_id"],
                "expected": row["expected"],
                "phase": "fresh",
                "execution_segment": "recovery",
            },
        )
        version = (result.get("gemini") or {}).get("reported_model_version")
        if version is not None and version != "gemini-3.8-flash":
            raise RuntimeError("Model version changed; stop rather than blend configurations")
    emit("completed", attempts=gemini.attempts, recovery=True)
    ledger.close()


if __name__ == "__main__":
    main()
