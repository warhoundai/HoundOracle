# ruff: noqa: F821
# This template inherits imports and helpers from cloud_compare_runner.py at build time.
"""Appended to the shared cloud runner by the builder; frozen supplemental paired experiment."""


def main():
    start = time.perf_counter()
    ledger = SpendLedger("/tmp/oracle-supplemental-spend.sqlite", authorized_usd=0.258048)
    config = VertexConfig(project=PAYLOAD["project"], response_format="evidence_indices")
    gemini = CountedGemini(VertexGemini(config, ledger))
    gemini.attempts = 35  # Already consumed in the first, completed execution.
    oracles = prepare_oracles()
    router = OracleFirstRouter(oracles["logistic"], gemini, minimum_confidence=0.85)
    emit("startup_stage", phase="ready", seconds=time.perf_counter() - start)
    rng = random.Random(1904)
    for repeat in range(3):
        for row in PAYLOAD["confirmation"]:
            order = ["baseline", "selected"]
            rng.shuffle(order)
            for arm in order:
                result = (
                    standalone(gemini, row)
                    if arm == "baseline"
                    else router.answer(row["question"], row["evidence"])
                )
                emit(
                    "query",
                    record={
                        "id": row["id"],
                        "group_id": row["group_id"],
                        "expected": row["expected"],
                        "repeat": repeat,
                        "arm": arm,
                        "phase": "supplemental",
                        "selected_policy": "precision",
                        **result,
                    },
                )
                version = (result.get("gemini") or {}).get("reported_model_version")
                if version is not None and version != "gemini-3.8-flash":
                    raise RuntimeError("Model version changed; stop comparison")
    emit(
        "completed",
        attempts=gemini.attempts,
        seconds=time.perf_counter() - start,
        selected="precision",
        response_format="evidence_indices",
    )
    ledger.close()


if __name__ == "__main__":
    main()
