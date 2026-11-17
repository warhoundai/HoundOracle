"""Paid-run caps, no gold leakage, safe continuation and uncertain usage accounting."""

import importlib.util
from pathlib import Path

import pytest
from test_experiments import fake_client

spec = importlib.util.spec_from_file_location(
    "public_runner", Path(__file__).resolve().parents[1] / "scripts/cloud_public_gemini_runner.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def payload(n=2, repetitions=3):
    return {
        "repetitions": repetitions,
        "questions": [
            {
                "id": str(i),
                "group_id": str(i),
                "question": "A factual claim.",
                "expected": "YES",
                "evidence": [{"url": "https://example.org/" + str(i), "text": "Evidence."}],
            }
            for i in range(n)
        ],
    }


def client_and_events():
    client = fake_client()
    events = []
    generate = client.models.generate_content

    def checked(**kwargs):
        assert any(kind == "reservation" for kind, _ in events)
        assert '"expected":' not in kwargs["contents"]
        result = generate(**kwargs)
        result.model_version = "gemini-3.8-flash"
        result.text = '{"decision":"MAYBE","evidence_indices":[]}'
        return result

    client.models.generate_content = checked
    return client, events


def test_new_authorization_and_workload_caps_are_required(tmp_path):
    client, events = client_and_events()
    with pytest.raises(PermissionError):
        runner.run(payload(), journal_path=tmp_path / "no.sqlite", authorized_usd=0, client=client)
    with pytest.raises(ValueError):
        runner.run(
            payload(49), journal_path=tmp_path / "many.sqlite", authorized_usd=5, client=client
        )
    with pytest.raises(ValueError):
        runner.run(
            payload(2, 4), journal_path=tmp_path / "repeats.sqlite", authorized_usd=5, client=client
        )
    assert client.models.calls == 0 and not events


def test_real_adapter_reserves_before_calls_and_never_repeats_completed_tasks(tmp_path):
    client, events = client_and_events()
    emit = lambda kind, **values: events.append((kind, values))
    result = runner.run(
        payload(), journal_path=tmp_path / "ok.sqlite", authorized_usd=5, client=client, emit=emit
    )
    assert result["summary"]["reserved_calls"] == 6 and client.models.calls == 12
    assert len(result["records"]) == 6
    runner.run(
        payload(), journal_path=tmp_path / "ok.sqlite", authorized_usd=5, client=client, emit=emit
    )
    assert client.models.calls == 12


def test_time_limit_and_failed_queries_are_not_falsehood_or_free_requests(tmp_path):
    client, events = client_and_events()
    emit = lambda kind, **values: events.append((kind, values))
    no_time = runner.run(
        payload(),
        journal_path=tmp_path / "time.sqlite",
        authorized_usd=5,
        client=client,
        emit=emit,
        seconds=0,
    )
    assert no_time["summary"]["reserved_calls"] == 0 and client.models.calls == 0

    def timeout(**kwargs):
        raise TimeoutError("fixture")

    client.models.generate_content = timeout
    result = runner.run(
        payload(1, 1),
        journal_path=tmp_path / "failure.sqlite",
        authorized_usd=5,
        client=client,
        emit=emit,
    )
    assert result["records"][0]["result"]["decision"] == "MAYBE"
    assert result["records"][0]["result"]["failure"] == "TimeoutError"
    assert result["summary"]["unknown_usage_calls"] == 1
    assert result["summary"]["token_cost_bounds_usd"][1] > 0


def test_definitive_response_exercises_citation_dependency(tmp_path):
    client = fake_client()
    generate = client.models.generate_content

    def definitive(**kwargs):
        result = generate(**kwargs)
        result.model_version = "gemini-3.8-flash"
        result.text = '{"decision":"YES","evidence_indices":[0]}'
        return result

    client.models.generate_content = definitive
    result = runner.run(
        payload(1, 1),
        journal_path=tmp_path / "yes.sqlite",
        authorized_usd=5,
        client=client,
        emit=lambda *args, **kwargs: None,
    )
    assert result["records"][0]["state"] == "SUCCEEDED"
    assert result["records"][0]["result"]["decision"] == "YES"
