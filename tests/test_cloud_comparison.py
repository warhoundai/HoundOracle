"""Exercise the full comparison/control loop without credentials or network calls."""

import importlib.util
import json
from pathlib import Path


def runner():
    path = Path(__file__).parents[1] / "scripts/cloud_compare_runner.py"
    spec = importlib.util.spec_from_file_location("comparison_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_regressing_oracle_is_bypassed_before_confirmation(monkeypatch, capsys):
    module = runner()

    class Ledger:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    class Gemini:
        def answer(self, question, evidence):
            return {
                "answer": {"decision": question},
                "token_cost_at_list_price_usd": 0.001,
                "output_valid": True,
            }

    class Oracle:
        def __init__(self, decision):
            self.decision = decision
            self.artifact = {"population": "fixture"}

        def verify_evidence(self, *args, **kwargs):
            return {
                "decision": self.decision,
                "confidence": 0.82 if self.decision == "YES" else None,
            }

    def row(i, label):
        return {
            "id": str(i),
            "group_id": "group" + str(i),
            "question": label,
            "expected": label,
            "evidence": [],
        }

    module.PAYLOAD = {
        "development": [row(i, label) for i, label in enumerate(["YES", "NO", "MAYBE"])],
        "confirmation": [
            row(i + 10, label) for i, label in enumerate(["YES", "NO", "MAYBE", "NO"])
        ],
    }
    monkeypatch.setattr(module, "SpendLedger", Ledger)
    monkeypatch.setattr(module, "choose_model", lambda ledger: module.CountedGemini(Gemini()))
    monkeypatch.setattr(
        module, "prepare_oracles", lambda: {"mlp": Oracle("YES"), "logistic": Oracle("MAYBE")}
    )
    module.main()
    logs = [json.loads(s) for s in capsys.readouterr().out.splitlines()]
    selection = next(r for r in logs if r["kind"] == "selection")
    assert selection["initial_worse"] and selection["selected"] == "bypass"
    confirmation = [
        r["record"] for r in logs if r["kind"] == "query" and r["record"]["phase"] == "confirmation"
    ]
    assert len(confirmation) == 16
    assert all(r["decision"] == r["expected"] for r in confirmation)
    assert all(r["oracle"] is None for r in confirmation if r["arm"] == "selected")
    assert next(r for r in logs if r["kind"] == "completed")["attempts"] <= 48


def test_global_call_cap_prevents_extra_inference():
    import pytest

    module = runner()

    class Gemini:
        def answer(self, *args, **kwargs):
            raise AssertionError("Must not call")

    counted = module.CountedGemini(Gemini())
    counted.attempts = 48
    with pytest.raises(PermissionError):
        counted.answer("x", [])


def test_analysis_does_not_call_partial_logs_complete():
    path = Path(__file__).parents[1] / "scripts/analyze_cloud_comparison.py"
    spec = importlib.util.spec_from_file_location("analysis_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = {
        "development": [],
        "confirmation": [{"id": "q", "expected": "MAYBE", "evidence": []}],
    }
    record = {
        "id": "q",
        "expected": "MAYBE",
        "decision": "MAYBE",
        "failure": "NETWORK_ERROR",
        "latency_seconds": 0.1,
        "token_cost_usd": None,
        "model_calls": 1,
        "phase": "confirmation",
        "arm": "baseline",
        "repeat": 0,
    }
    raw = [
        {"campaign": "oracle-cloud-03", "kind": "completed"},
        {"campaign": "oracle-cloud-03", "kind": "query", "record": record},
    ]
    result = module.analyze(raw, payload)
    assert result["complete"] is False
    assert result["systems"]["confirmation:baseline"]["accuracy"] == 0
    assert result["systems"]["confirmation:baseline"]["token_list_price_cost_usd"] is None
    import pytest

    with pytest.raises(ValueError, match="Duplicate"):
        module.analyze(raw + [raw[-1]], payload)


def test_unknown_provider_usage_and_timing_stay_unknown():
    path = Path(__file__).parents[1] / "scripts/analyze_cloud_comparison.py"
    spec = importlib.util.spec_from_file_location("analysis_uncertainty", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {
            "id": str(i),
            "decision": "YES",
            "expected": "YES",
            "latency_seconds": float(i + 1),
            "token_cost_usd": 0.001,
            "model_calls": 1,
        }
        for i in range(20)
    ]
    rows[0].update(
        decision="MAYBE",
        failure="PROVIDER_TIMEOUT",
        latency_seconds=None,
        token_cost_usd=None,
        reserved_usd=0.02,
    )
    report = module.summarize(rows)
    assert report["accuracy"] == 0.95 and report["failed_queries"] == 1
    assert report["p95_seconds"] is None and report["unknown_latency_count"] == 1
    assert report["latency_quantile_bounds_seconds"]["p95"][0] <= 19.5
    assert report["latency_quantile_bounds_seconds"]["p95"][1] is None  # Upper bound is unbounded.
    assert report["token_list_price_cost_usd"] is None
    assert abs(report["token_cost_total_bounds_usd"][0] - 0.019) < 1e-10
    assert abs(report["token_cost_total_bounds_usd"][1] - 0.039) < 1e-10
