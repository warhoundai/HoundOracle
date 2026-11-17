"""Executed only by the prepared temporary Cloud Run Job. No ground-truth labels enter inference."""

import json
import random
import time
from datetime import UTC
from pathlib import Path

from houndoracle.experiments.head import HeadNLI
from houndoracle.experiments.nli import OnnxNLI
from houndoracle.experiments.pipeline import ResearchPipeline
from houndoracle.experiments.router import OracleFirstRouter
from houndoracle.experiments.vertex import SpendLedger, VertexConfig, VertexGemini

# Injected by builder: PAYLOAD has external labels only for scoring/selection.
PAYLOAD = globals().get("PAYLOAD", {})


def emit(kind, **values):
    print(json.dumps({"campaign": "oracle-cloud-03", "kind": kind, **values}), flush=True)


class CountedGemini:
    def __init__(self, delegate):
        self.delegate = delegate
        self.attempts = 0

    def answer(self, *args, **kwargs):
        if self.attempts >= 48:
            raise PermissionError("48-attempt global cap reached")
        self.attempts += 1
        return self.delegate.answer(*args, **kwargs)


def choose_model(ledger):
    attempts = 0
    for name in ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-2.5-flash"]:
        prices = (0.3, 2.5) if name == "gemini-2.5-flash" else (0.75, 3.75)
        config = VertexConfig(
            project=PAYLOAD["project"],
            model=name,
            max_output_tokens=4096,
            input_usd_per_million=prices[0],
            output_usd_per_million=prices[1],
            thinking_budget=1024 if name == "gemini-2.5-flash" else None,
        )
        try:
            client = VertexGemini(config, ledger)
            attempts += 1
            smoke = client.answer(
                "Two plus two equals four.",
                [{"url": "https://example.org/arithmetic", "text": "2 + 2 = 4."}],
            )
            emit("model_selected", model=name, smoke=smoke, attempts=attempts)
            result = CountedGemini(client)
            result.attempts = attempts
            return result
        except Exception as exc:
            code = getattr(exc, "code", None)
            emit("model_unavailable", model=name, error_type=type(exc).__name__, code=code)
            if code == 403:
                raise
    raise RuntimeError("No permitted Flash model succeeded")


def prepare_oracles():
    import urllib.error
    import urllib.request
    from datetime import datetime
    from email.utils import parsedate_to_datetime

    directory = Path("/tmp/oracle-model")
    directory.mkdir(exist_ok=True)
    for filename in ["config.json", "tokenizer.json", "onnx/model.onnx"]:
        target = directory / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        url = (
            "https://huggingface.co/"
            + PAYLOAD["model_manifest"]["model_id"]
            + "/resolve/"
            + PAYLOAD["model_manifest"]["revision"]
            + "/"
            + filename
        )
        try:
            response = urllib.request.urlopen(url, timeout=45)
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise
            value = exc.headers.get("Retry-After", "60")
            try:
                delay = float(value)
            except ValueError:
                delay = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
            if delay > 60:
                raise RuntimeError("Model provider Retry-After exceeds bounded recovery") from exc
            time.sleep(max(0, delay))
            response = urllib.request.urlopen(url, timeout=45)
        with response, target.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    (directory / "manifest.json").write_text(json.dumps(PAYLOAD["model_manifest"]))
    base = OnnxNLI(directory, variant="fp32", threads=PAYLOAD.get("threads", 1))
    result = {}
    for name in PAYLOAD.get("oracle_heads", ["mlp", "logistic"]):
        result[name] = ResearchPipeline(
            HeadNLI(base, PAYLOAD["heads"][name], name), PAYLOAD["calibrations"][name]
        )
    return result


def aggregate(records):
    if not records:
        return {}
    ordered = sorted(r["latency_seconds"] for r in records)
    return {
        "count": len(records),
        "accuracy": sum(r["decision"] == r["expected"] and not r.get("failure") for r in records)
        / len(records),
        "coverage": sum(r["decision"] in ("YES", "NO") for r in records) / len(records),
        "token_cost_usd": sum(r["token_cost_usd"] for r in records)
        if all(r["token_cost_usd"] is not None for r in records)
        else None,
        "model_calls": sum(r["model_calls"] for r in records),
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
    }


def standalone(gemini, row):
    started = time.perf_counter()
    result = gemini.answer(row["question"], row["evidence"])
    return {
        "decision": result["answer"]["decision"],
        "gemini": result,
        "oracle": None,
        "source": "gemini_standalone",
        "latency_seconds": time.perf_counter() - started,
        "token_cost_usd": result["token_cost_at_list_price_usd"],
        "model_calls": 1,
        "failure": None
        if result.get("output_valid", True)
        else result.get("failure_reason") or "INVALID_MODEL_OUTPUT",
    }


def main():
    start = time.perf_counter()
    ledger = SpendLedger("/tmp/oracle-cloud-spend.sqlite", authorized_usd=2.50)
    gemini = choose_model(ledger)
    emit("startup_stage", phase="before_oracle_model_load", seconds=time.perf_counter() - start)
    oracles = prepare_oracles()
    emit("startup_stage", phase="ready", seconds=time.perf_counter() - start)
    routers = {
        "mlp": OracleFirstRouter(oracles["mlp"], gemini),
        "precision": OracleFirstRouter(oracles["logistic"], gemini, minimum_confidence=0.85),
        "bypass": OracleFirstRouter(oracles["mlp"], gemini, bypass=True),
    }
    records = {"baseline": [], "mlp": []}
    rng = random.Random(1729)
    for repeat in range(2):
        for row in PAYLOAD["development"]:
            order = ["baseline", "mlp"]
            rng.shuffle(order)
            for arm in order:
                result = (
                    standalone(gemini, row)
                    if arm == "baseline"
                    else routers[arm].answer(row["question"], row["evidence"])
                )
                record = {
                    "id": row["id"],
                    "group_id": row["group_id"],
                    "expected": row["expected"],
                    "repeat": repeat,
                    "arm": arm,
                    "phase": "development",
                    **result,
                }
                records[arm].append(record)
                emit("query", record=record)
    baseline = aggregate(records["baseline"])
    initial = aggregate(records["mlp"])
    worse = (
        initial["accuracy"] < baseline["accuracy"]
        or initial["coverage"] < baseline["coverage"]
        or initial["token_cost_usd"] is None
        or baseline["token_cost_usd"] is None
        or initial["token_cost_usd"] > baseline["token_cost_usd"]
        or initial["p95"] > baseline["p95"]
    )
    selected = "mlp"
    if worse:
        # Development-only replay screens an existing higher-precision model, never retrains on confirmation.
        lookup = {(r["id"], r["repeat"]): r for r in records["baseline"]}
        replay = []
        for repeat in range(2):
            for row in PAYLOAD["development"]:
                prediction = oracles["logistic"].verify_evidence(
                    row["question"],
                    row["evidence"],
                    population=oracles["logistic"].artifact["population"],
                    use_cache=False,
                )
                accepted = (
                    prediction["decision"] in ("YES", "NO")
                    and (prediction["confidence"] or 0) >= 0.85
                )
                base = lookup[(row["id"], repeat)]
                replay.append(
                    {
                        **base,
                        "decision": prediction["decision"] if accepted else base["decision"],
                        "model_calls": 0 if accepted else 1,
                        "token_cost_usd": 0 if accepted else base["token_cost_usd"],
                        "failure": None if accepted else base.get("failure"),
                    }
                )
        candidate = aggregate(replay)
        selected = (
            "precision"
            if (
                candidate["accuracy"] >= baseline["accuracy"]
                and candidate["coverage"] >= baseline["coverage"]
                and candidate["model_calls"] < baseline["model_calls"]
                and candidate["token_cost_usd"] is not None
                and baseline["token_cost_usd"] is not None
                and candidate["token_cost_usd"] <= baseline["token_cost_usd"]
            )
            else "bypass"
        )
        emit(
            "development_replay", policy="precision", summary=candidate, live_latency_measured=False
        )
    emit(
        "selection",
        selected=selected,
        initial_worse=worse,
        baseline=baseline,
        initial=initial,
        rule="Freeze before fresh confirmation; no confirmation-based tuning",
    )
    for repeat in range(2):
        for row in PAYLOAD["confirmation"]:
            order = ["baseline", "selected"]
            rng.shuffle(order)
            for arm in order:
                result = (
                    standalone(gemini, row)
                    if arm == "baseline"
                    else routers[selected].answer(row["question"], row["evidence"])
                )
                emit(
                    "query",
                    record={
                        "id": row["id"],
                        "group_id": row["group_id"],
                        "expected": row["expected"],
                        "repeat": repeat,
                        "arm": arm,
                        "selected_policy": selected,
                        "phase": "confirmation",
                        **result,
                    },
                )
    emit(
        "completed",
        attempts=gemini.attempts,
        seconds=time.perf_counter() - start,
        selected=selected,
    )
    ledger.close()


if __name__ == "__main__":
    main()
