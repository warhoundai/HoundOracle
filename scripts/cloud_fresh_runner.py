# ruff: noqa: F821
# Appended to shared cloud_compare_runner.py helpers by the manifest builder.
"""Frozen larger matched-evidence comparison. Never tunes on the fresh cohort."""


def main():
    ledger = SpendLedger("/tmp/fresh-spend.sqlite", authorized_usd=6.193152)
    config = VertexConfig(project=PAYLOAD["project"], seed=1907)
    gemini = FreshCounter(VertexGemini(config, ledger))
    # This campaign has a separate explicit 288-call limit, not the old pilot allowance.
    oracles = prepare_oracles()
    router = OracleFirstRouter(
        oracles["logistic"],
        gemini,
        minimum_confidence=0.85,
        complete_evidence=True,
        evaluation_cache_size=0,
    )
    emit(
        "configuration",
        config=__import__("dataclasses").asdict(config),
        policy="fixed-logistic-.85-cold-cache",
        authorized_call_cap=288,
    )
    rng = random.Random(1907)
    for repeat in range(3):
        for row in PAYLOAD["confirmation"]:
            order = ["baseline", "selected"]
            rng.shuffle(order)
            for arm in order:
                # Identical evidence-only preflight benefits both systems.
                preflight_started = time.perf_counter()
                before = gemini.attempts
                try:
                    if arm == "baseline" and row["evidence"] == []:
                        result = {
                            "decision": "MAYBE",
                            "gemini": None,
                            "oracle": None,
                            "source": "empty_permitted_corpus",
                            "latency_seconds": time.perf_counter() - preflight_started,
                            "token_cost_usd": 0.0,
                            "model_calls": 0,
                            "failure": None,
                        }
                    else:
                        result = (
                            standalone(gemini, row)
                            if arm == "baseline"
                            else router.answer(
                                row["question"], row["evidence"], proposition_validated=True
                            )
                        )
                except Exception as exc:  # noqa: BLE001 -- every attempted task must retain a failure record
                    result = {
                        "decision": "MAYBE",
                        "gemini": None,
                        "oracle": None,
                        "source": "provider_or_evaluator_failure",
                        "latency_seconds": time.perf_counter() - preflight_started,
                        "token_cost_usd": None,
                        "model_calls": gemini.attempts - before,
                        "failure": type(exc).__name__,
                        "provider_code": getattr(exc, "code", None),
                        "reserved_usd": (gemini.attempts - before) * config.maximum_model_cost,
                    }
                emit(
                    "query",
                    record={
                        **result,
                        "id": row["id"],
                        "group_id": row["group_id"],
                        "expected": row["expected"],
                        "arm": arm,
                        "repeat": repeat,
                        "phase": "fresh",
                    },
                )
                version = (result.get("gemini") or {}).get("reported_model_version")
                if version is not None and version != "gemini-3.8-flash":
                    raise RuntimeError("Frozen model version changed")
    emit("completed", attempts=gemini.attempts)
    ledger.close()


class FreshCounter:
    def __init__(self, delegate):
        self.delegate = delegate
        self.attempts = 0

    def answer(self, *args, **kwargs):
        if self.attempts >= 288:
            raise PermissionError("Fresh campaign call cap exhausted")
        self.attempts += 1
        return self.delegate.answer(*args, **kwargs)


if __name__ == "__main__":
    main()
