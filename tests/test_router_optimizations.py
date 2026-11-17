"""Safety boundaries for opt-in inference avoidance and evaluator reuse."""

import pytest

from houndoracle.experiments.router import OracleFirstRouter


class Oracle:
    def __init__(self):
        self.calls = 0
        self.artifact = {"population": "test", "version": 1}

    def verify_evidence(self, *args, **kwargs):
        self.calls += 1
        return {"decision": "YES", "confidence": 0.9, "facts": ["original"]}


class Gemini:
    def __init__(self):
        self.calls = 0

    def answer(self, *args):
        self.calls += 1
        return {"answer": {"decision": "MAYBE"}, "token_cost_at_list_price_usd": 0.001}


def test_empty_shortcut_requires_complete_corpus_and_validated_proposition():
    model, gemini = Oracle(), Gemini()
    router = OracleFirstRouter(model, gemini, complete_evidence=True)
    result = router.answer("A claim", [], proposition_validated=True)
    assert result["decision"] == "MAYBE" and result["confidence"] is None
    assert model.calls == gemini.calls == 0
    assert router.answer("A claim", [])["source"] == "oracle"
    assert (
        OracleFirstRouter(model, gemini).answer("A claim", [], proposition_validated=True)["source"]
        == "oracle"
    )
    assert (
        OracleFirstRouter(model, gemini, complete_evidence=True, bypass=True).answer(
            "A claim", [], proposition_validated=True
        )["source"]
        == "gemini_fallback"
    )


def test_cache_exactness_expiry_isolation_and_copy():
    model, gemini = Oracle(), Gemini()
    now = [0.0]
    router = OracleFirstRouter(
        model,
        gemini,
        evaluation_cache_size=2,
        permission_scope="tenant:a",
        evaluation_cache_ttl=2,
        clock=lambda: now[0],
    )
    evidence = [{"url": "https://example.org", "text": "Fact", "revision": 1}]
    result = router.answer("Claim in 2024", evidence)
    result["oracle"]["facts"].append("poison")
    hit = router.answer("Claim in 2024", evidence)
    assert hit["evaluation_cache_hit"] and hit["oracle"]["facts"] == ["original"]
    for question in ["Claim in 2025", "Not claim in 2024"]:
        assert not router.answer(question, evidence)["evaluation_cache_hit"]
    assert len(router.cache) <= 2
    assert not router.answer("Claim in 2024", evidence)["evaluation_cache_hit"]  # LRU eviction
    evidence[0]["revision"] = 2
    assert not router.answer("Claim in 2024", evidence)["evaluation_cache_hit"]
    model.artifact["version"] = 2
    assert not router.answer("Claim in 2024", evidence)["evaluation_cache_hit"]
    router.permission_scope = "tenant:b"
    assert not router.answer("Claim in 2024", evidence)["evaluation_cache_hit"]
    now[0] = 2
    assert not router.answer("Claim in 2024", evidence)["evaluation_cache_hit"]
    assert gemini.calls == 0


def test_no_gemini_answer_reuse():
    model, gemini = Oracle(), Gemini()
    model.verify_evidence = lambda *a, **k: {"decision": "MAYBE", "confidence": None}
    router = OracleFirstRouter(model, gemini, evaluation_cache_size=2, permission_scope="a")
    router.answer("Claim", [])
    assert router.answer("Claim", [])["evaluation_cache_hit"]
    assert gemini.calls == 2
    with pytest.raises(ValueError):
        OracleFirstRouter(model, gemini, evaluation_cache_size=1)
    with pytest.raises(ValueError):
        OracleFirstRouter(model, gemini, evaluation_cache_ttl=float("nan"))
