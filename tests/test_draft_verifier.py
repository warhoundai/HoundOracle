import json

import pytest

from houndoracle.experiments.draft_verifier import predict


def response(text):
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 3},
    }


def test_draft_is_not_evidence_and_is_not_exposed_as_public_result():
    calls = []
    draft = "Private analysis of the supplied passage."

    def generate(**kwargs):
        calls.append(kwargs)
        return response(
            draft if kwargs["phase"] == "analysis" else '{"decision":"YES","evidence_indices":[0]}'
        )

    seen = []
    result = predict(
        generate,
        "X is not Y",
        [{"text": "X differs from Y"}],
        mode="reasoning-first",
        observer=seen.append,
    )
    assert result["candidate"] == "YES" and len(calls) == 2
    assert all(c["messages"][0]["role"] == "user" for c in calls)
    assert calls[1]["messages"][1]["content"] == draft
    assert seen == [draft] and draft not in json.dumps(result)
    assert result["calibrated_correctness"] is None and not result["production_qualified"]


def test_direct_and_empty_evidence_have_bounded_call_counts():
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return response('{"decision":"MAYBE","evidence_indices":[]}')

    predict(generate, "P", [{"text": "Q"}], mode="direct")
    assert len(calls) == 1
    predict(generate, "P", [], mode="reasoning-first")
    assert len(calls) == 1
    with pytest.raises(ValueError):
        predict(generate, "P", [], mode="unknown")


def test_missing_draft_is_a_failure_without_formatting_a_fake_answer():
    result = predict(lambda **kwargs: response(None), "P", [{"text": "Q"}], mode="reasoning-first")
    assert result["candidate"] == "MAYBE" and result["failure"] == "EMPTY_DRAFT"
