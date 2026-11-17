import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_experiments import fake_client

spec = importlib.util.spec_from_file_location(
    "hybrid_runner", Path(__file__).parents[1] / "scripts/cloud_hybrid_workflow_runner.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Local:
    model = SimpleNamespace(config={"test": "frozen"})

    def __init__(self):
        self.cache = set()

    def verify_evidence(self, question, evidence, *, population, use_cache):
        hit = use_cache and question in self.cache
        if use_cache:
            self.cache.add(question)
        return {
            "decision": "YES" if question == "q0" else "MAYBE",
            "cache_hit": hit,
            "confidence": 0.9 if question == "q0" else None,
        }


def data():
    return {
        "questions": [
            {
                "id": f"q{i}",
                "group_id": str(i),
                "question": f"q{i}",
                "evidence": [{"url": "https://example.org", "text": "An evidence passage."}],
                "expected": "YES" if i == 0 else "MAYBE",
            }
            for i in range(2)
        ],
        "repetitions": 3,
        "calibration": {"population": "test"},
        "frozen_local_decisions": {"q0": "YES", "q1": "MAYBE"},
        "dataset_sha256": "frozen",
    }


def test_actual_fallback_and_cache_avoid_only_qualified_calls(tmp_path):
    client = fake_client()
    generate = client.models.generate_content

    def response(**kwargs):
        assert '"expected":' not in kwargs["contents"]
        result = generate(**kwargs)
        result.text = '{"decision":"MAYBE","evidence_indices":[]}'
        result.model_version = "gemini-3.8-flash"
        return result

    client.models.generate_content = response
    result = runner.run(
        data(),
        authorized_usd=8,
        model=Local(),
        client=client,
        emit=lambda *a, **k: None,
        journal_path=tmp_path / "run.sqlite",
    )
    assert len(result["records"]) == 12
    assert result["summary"]["reserved_calls"] == 9
    assert client.models.calls == 18
    hybrid = [r for r in result["records"] if r["arm"] == "hybrid"]
    assert sum(r["result"]["local"]["cache_hit"] for r in hybrid) == 4
    assert sum(r["result"]["source"] == "oracle" for r in hybrid) == 3
    with pytest.raises(PermissionError):
        runner.run(data(), authorized_usd=0, model=Local(), client=client)


def test_policy_mismatch_stops_before_any_paid_request(tmp_path):
    payload = data()
    payload["frozen_local_decisions"]["q0"] = "NO"
    client = fake_client()
    with pytest.raises(ValueError, match="frozen local decision"):
        runner.run(
            payload,
            authorized_usd=8,
            model=Local(),
            client=client,
            journal_path=tmp_path / "bad.sqlite",
        )
    assert client.models.calls == 0
