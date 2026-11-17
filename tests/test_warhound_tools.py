import json

import pytest

from houndoracle.budget import Budget, BudgetExceeded
from houndoracle.network import RetrievalError
from houndoracle.warhound import HoundBrowserAdapter, HoundIndexAdapter, HoundSourceAdapter
from houndoracle.web import SearXNGAdapter


def test_scraper_uses_published_schema_and_never_trusts_source(monkeypatch):
    def transport(url, **kw):
        assert url == "http://127.0.0.1:8081/v1/scrape"
        body = json.loads(kw["body"])
        assert body["options"]["respect_robots"] and body["options"]["render"] == "auto"
        assert body["options"]["formats"] == ["text"]
        assert "prompt" not in body["options"] and "actions" not in body["options"]
        assert kw["headers"]["Authorization"] == "Bearer local-secret"
        return {
            "body": json.dumps(
                {
                    "final_url": body["url"],
                    "data": {"text": "A release record"},
                    "acquisition": "browser",
                    "reviewed": True,
                }
            ).encode()
        }

    monkeypatch.setattr("houndoracle.warhound.addresses", lambda *a: ["93.184.216.34"])
    monkeypatch.setattr("houndoracle.warhound.fetch", transport)
    adapter = HoundSourceAdapter("http://127.0.0.1:8081", "local-secret", render="auto")
    row = adapter("https://example.org/release", Budget(1, 20, 0))
    assert row["text"] == "A release record" and not row["reviewed"]
    assert row["retrieval_provider"] == "houndcrawler"
    assert "local-secret" not in repr(adapter)


def test_challenge_is_quarantined_not_ingested(monkeypatch):
    monkeypatch.setattr("houndoracle.warhound.addresses", lambda *a: ["93.184.216.34"])
    monkeypatch.setattr(
        "houndoracle.warhound.fetch",
        lambda *a, **k: {
            "body": json.dumps(
                {"url": "https://example.org/", "data": {"text": "Verify you are human"}}
            ).encode()
        },
    )
    with pytest.raises(RetrievalError, match="CAPTCHA_REQUIRES_INTERACTION"):
        HoundSourceAdapter("http://127.0.0.1:8081", "key")("https://example.org/", Budget(1, 10, 0))


def test_houndsearch_contract_bounds_upstream_calls(monkeypatch):
    def transport(url, **kw):
        payload = json.loads(kw["body"])
        assert payload["budget"]["max_requests"] == 1
        assert not payload["include_contents"] and payload["mode"] == "fast"
        return {
            "body": json.dumps(
                {
                    "results": [
                        {
                            "url": "https://example.org/release",
                            "snippet": "Snippet",
                            "evidence": [{"text": "Exact indexed source passage"}],
                        }
                    ]
                }
            ).encode()
        }

    monkeypatch.setattr("houndoracle.warhound.fetch", transport)
    rows = HoundIndexAdapter("http://127.0.0.1:8082", "key").retrieve("release", Budget(1, 10, 0))
    assert rows[0]["text"] == "Exact indexed source passage"
    assert rows[0]["retrieval_metadata"]["kind"] == "source_document"


def test_browser_closes_session_and_never_replays_unknown_commands(monkeypatch):
    monkeypatch.setattr("houndoracle.web.check_source_policy", lambda url, budget: url)
    calls = []

    def transport(url, **kw):
        calls.append((url, kw))
        if url.endswith("/sessions"):
            data = {"id": "test-session", "revision": 0}
        elif kw["method"] == "DELETE":
            data = {}
        elif url.endswith("/commands"):
            data = {"id": "accepted"}
            assert json.loads(kw["body"])["action"] == "navigate"
        else:
            data = {"status": "outcome_unknown"}
        return {"body": json.dumps(data).encode()}

    monkeypatch.setattr("houndoracle.warhound.fetch", transport)
    client = HoundBrowserAdapter("http://127.0.0.1:8084", "key")
    with pytest.raises(RetrievalError, match="BROWSER_OUTCOME_UNKNOWN"):
        client.observe("https://example.org/", Budget(4, 10, 0))
    assert len(calls) == 4 and calls[-1][1]["method"] == "DELETE"
    calls.clear()
    with pytest.raises(BudgetExceeded):
        client.observe("https://example.org/", Budget(1, 10, 0))
    assert calls == []


def test_google_language_and_challenge_cooldown_with_working_fallback(monkeypatch):
    from urllib.parse import parse_qs, urlsplit

    requests = []

    def transport(url, **kw):
        query = parse_qs(urlsplit(url).query)
        requests.append(query)
        assert query["language"] == ["en-US"]
        failures = (
            [["google", "access denied"], ["duckduckgo", "CAPTCHA"]] if len(requests) == 1 else []
        )
        return {
            "body": json.dumps(
                {
                    "results": [{"url": "https://example.org/", "content": "Source"}],
                    "unresponsive_engines": failures,
                }
            ).encode()
        }

    monkeypatch.setattr("houndoracle.web.fetch", transport)
    adapter = SearXNGAdapter("http://127.0.0.1:8080", fetch_sources=False)
    assert adapter.retrieve("release", Budget(1, 10, 0))
    assert adapter.last_status["failures"] == [
        {"engine": "google", "code": "ACCESS_DENIED"},
        {"engine": "duckduckgo", "code": "CAPTCHA_REQUIRED"},
    ]
    assert adapter.retrieve("release", Budget(1, 10, 0))
    assert requests[1]["engines"] == ["bing"]


def test_durable_crawl_is_bounded_idempotent_and_importable(monkeypatch):
    monkeypatch.setattr("houndoracle.warhound.addresses", lambda *a: ["93.184.216.34"])

    def transport(url, **kw):
        if url.endswith("/jobs"):
            payload = json.loads(kw["body"])
            assert payload["max_pages"] == 3 and payload["max_concurrency"] == 1
            assert payload["max_seconds"] == 60 and not payload["allow_external_links"]
            assert len(kw["headers"]["Idempotency-Key"]) == 32
            return {"body": json.dumps({"id": "a" * 32}).encode()}
        assert url.endswith("/results?cursor=0&limit=20")
        return {
            "body": json.dumps(
                {
                    "data": [
                        {
                            "result": {
                                "url": "https://example.org/",
                                "data": {"text": "Source record"},
                            }
                        },
                        {"result": {"url": "http://127.0.0.1/", "data": {"text": "Private"}}},
                        {
                            "result": {
                                "url": "https://other.example.org/",
                                "data": {"text": "Verify you are human"},
                            }
                        },
                    ],
                    "next_cursor": 20,
                }
            ).encode()
        }

    monkeypatch.setattr("houndoracle.warhound.fetch", transport)
    crawler = HoundSourceAdapter("http://127.0.0.1:8081", "key")
    assert (
        crawler.submit_crawl("https://example.org/", Budget(1, 20, 0), max_pages=3)["id"]
        == "a" * 32
    )
    page = crawler.crawl_results("a" * 32, Budget(1, 20, 0))
    assert len(page["results"]) == 1 and page["next_cursor"] == 20 and not page["verified"]


def test_real_browser_http_created_accepted_and_no_content_responses(monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    monkeypatch.setattr("houndoracle.web.check_source_policy", lambda url, budget: url)
    methods = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.headers["Authorization"] == "Bearer instance-key"
            methods.append("POST")
            if self.path.endswith("/sessions"):
                assert body["scope"]["methods"] == ["GET", "HEAD"]
                code, payload = 201, {"id": "session1", "revision": 0}
            else:
                code, payload = 202, {"id": body["id"]}
            self.send_response(code)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def do_GET(self):
            methods.append("GET")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"completed","result":{"text":"Release record"}}')

        def do_DELETE(self):
            methods.append("DELETE")
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = HoundBrowserAdapter(f"http://127.0.0.1:{server.server_port}", "instance-key")
        result = client.observe("https://example.org/", Budget(4, 10, 0))
        assert result["status"] == "OBSERVED" and not result["verified"]
        assert methods == ["POST", "POST", "GET", "DELETE"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
