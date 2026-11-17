import json

import pytest

from houndoracle import Oracle
from houndoracle.budget import Budget, BudgetExceeded
from houndoracle.network import RetrievalError
from houndoracle.search import (
    ExaSearchAdapter,
    FederatedSearch,
    WikipediaSearchAdapter,
    provider_from_environment,
    source_url,
)


def response(rows):
    return {
        "body": json.dumps(
            {"results": rows, "requestId": "test", "costDollars": {"total": 0.007}}
        ).encode()
    }


def test_exa_reserves_before_io_and_does_not_trust_or_duplicate_results(monkeypatch):
    budget = Budget(1, 10, 0.02)

    def transport(url, **options):
        assert budget.requests == 1 and budget.cost == 0.02
        assert url == "https://api.exa.ai/search"
        assert options["headers"]["x-api-key"] == "test-secret"
        body = json.loads(options["body"])
        assert body["contents"]["text"]["maxCharacters"] == 20000 and body["numResults"] == 3
        return response(
            [
                {
                    "url": "https://docs.example.org/a?utm_source=x",
                    "text": "A primary-looking claim.",
                    "reviewed": True,
                },
                {"url": "https://docs.example.org/a", "text": "Different content"},
                {"url": "https://other.example.org/b", "text": "A primary-looking claim."},
                {"url": "http://169.254.169.254/latest", "text": "Private data"},
            ]
        )

    monkeypatch.setattr("houndoracle.search.fetch", transport)
    result = ExaSearchAdapter("test-secret").retrieve("release date", budget)
    assert len(result) == 1 and result[0]["reviewed"] is False and result[0]["primary"] is False
    assert result[0]["url"] == "https://docs.example.org/a"
    assert result[0]["retrieval_metadata"]["reported_cost_usd"] == 0.007
    assert "test-secret" not in repr(ExaSearchAdapter("test-secret"))
    with pytest.raises(BudgetExceeded):
        ExaSearchAdapter("test-secret").retrieve("q", budget)


def test_search_requires_permission_and_budget_then_reuses_only_evidence(monkeypatch):
    calls = []

    class Provider:
        name = "test"

        def retrieve(self, query, budget):
            budget.reserve(0.02)
            calls.append(query)
            return [
                {
                    "url": "https://example.org/release",
                    "text": "Widget was released in 2024.",
                    "reviewed": True,
                }
            ]

    oracle = Oracle(provider=Provider(), local_only=False)
    with pytest.raises(PermissionError):
        oracle.web_search("Widget")
    denied = oracle.web_search("Widget", allow_network=True, max_cost_usd=0)
    assert denied["reason"] == "BUDGET_EXHAUSTED" and calls == []
    result = oracle.web_search("Widget", allow_network=True, max_cost_usd=0.02)
    assert result["verified"] is False and len(calls) == 1
    hit = oracle.web_search("Widget", allow_network=False)
    assert hit["source"] == "local" and len(calls) == 1
    decision = oracle.verify("Was Widget released in 2024?")
    assert decision.decision.value == "MAYBE"
    assert all(json.loads(d["data"])["reviewed"] is False for d in oracle.store.documents())
    oracle.store.close()


def test_fallback_works_but_failure_is_not_evidence_of_falsehood():
    class Broken:
        name = "exa"

        def retrieve(self, q, budget):
            budget.reserve(0.02)
            raise RetrievalError("HTTP_429")

    class Backup:
        name = "backup"

        def retrieve(self, q, budget):
            budget.reserve(0)
            return [{"url": "https://example.org", "text": "A source."}]

    budget = Budget(2, 10, 0.02)
    result = FederatedSearch([Broken(), Backup()]).retrieve("q", budget)
    assert len(result) == 1 and budget.requests == 2
    with pytest.raises(RetrievalError, match="ALL_SEARCH_PROVIDERS_FAILED"):
        FederatedSearch([Broken()]).retrieve("q", Budget(1, 10, 0.02))


def test_wikipedia_no_model_call_and_private_sources_rejected(monkeypatch):
    monkeypatch.setattr(
        "houndoracle.search.fetch",
        lambda *a, **k: {
            "body": json.dumps(
                {
                    "query": {
                        "pages": [
                            {
                                "index": 1,
                                "fullurl": "https://en.wikipedia.org/wiki/Python",
                                "extract": "An encyclopedia extract.",
                                "title": "Python",
                            }
                        ]
                    }
                }
            ).encode()
        },
    )
    budget = Budget(1, 10, 0)
    assert WikipediaSearchAdapter().retrieve("Python", budget)[0]["reviewed"] is False
    assert budget.requests == 1 and budget.cost == 0
    for url in [
        "http://localhost/x",
        "http://127.0.0.1",
        "http://10.0.0.1",
        "https://user:pass@example.org",
        "http://metadata.internal/x",
        "https://example.org:9000",
    ]:
        assert source_url(url) is None


def test_key_file_owner_only_and_factory_has_no_network_side_effects(tmp_path, monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    p = tmp_path / "key"
    p.write_text("private-token")
    p.chmod(0o644)
    monkeypatch.setenv("EXA_API_KEY_FILE", str(p))
    with pytest.raises(PermissionError):
        provider_from_environment()
    p.chmod(0o600)
    provider = provider_from_environment()
    assert provider.providers[0].name == "exa"


def test_rest_online_search_and_offline_default(monkeypatch):
    from fastapi.testclient import TestClient

    from houndoracle.api import create_app

    class Provider:
        name = "test"

        def retrieve(self, q, budget):
            budget.reserve(0.02)
            return [{"url": "https://example.org/a", "text": "Search text."}]

    oracle = Oracle(provider=Provider(), local_only=False)
    with TestClient(create_app(oracle, api_key="x" * 32)) as client:
        headers = {"Authorization": "Bearer " + "x" * 32}
        assert (
            client.post("/v1/search", headers=headers, json={"query": "anything"}).json()["results"]
            == []
        )
        result = client.post(
            "/v1/search",
            headers=headers,
            json={"query": "anything", "allow_network": True, "max_cost_usd": 0.02},
        ).json()
        assert result["verified"] is False and len(result["results"]) == 1
        assert result["usage"]["requests"] == 1
    oracle.store.close()


def test_fixed_search_api_transport_blocks_other_origins_and_redirects(monkeypatch):
    import urllib.request

    from houndoracle.network import fetch_search_api

    with pytest.raises(RetrievalError, match="UNSAFE_SEARCH_ENDPOINT"):
        fetch_search_api("https://attacker.example/search", headers={"x-api-key": "secret"})

    class Opener:
        def open(self, request, **kwargs):
            raise RetrievalError("ADAPTER_REDIRECT_BLOCKED")

    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: Opener())
    with pytest.raises(RetrievalError, match="ADAPTER_REDIRECT_BLOCKED"):
        fetch_search_api(
            "https://api.exa.ai/search", method="POST", headers={"x-api-key": "secret"}
        )
