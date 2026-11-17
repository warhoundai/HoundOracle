import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from houndoracle import Oracle
from houndoracle.budget import Budget
from houndoracle.network import RetrievalError
from houndoracle.web import SearXNGAdapter, acquire_source


def test_local_search_service_real_http_and_local_evidence_reuse(monkeypatch):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            params = parse_qs(urlsplit(self.path).query)
            calls.append(params)
            assert params["engines"] == ["google,bing,duckduckgo"]
            assert params["format"] == ["json"]
            body = json.dumps(
                {
                    "results": [
                        {
                            "url": "https://www.python.org/downloads/release/python-3120/",
                            "title": "Python release",
                            "content": "<b>Python</b> was released in 2023.",
                            "engines": ["google", "bing"],
                        },
                        {"url": "http://169.254.169.254/latest", "content": "private"},
                    ]
                }
            ).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    oracle = Oracle(
        local_only=False,
        provider=SearXNGAdapter(f"http://127.0.0.1:{server.server_port}", fetch_sources=False),
    )
    try:
        with pytest.raises(PermissionError):
            oracle.web_search("Python")
        result = oracle.web_search("Python", allow_network=True)
        assert len(result["results"]) == 1
        assert result["usage"]["requests"] == 1
        assert result["usage"]["reserved_cost_usd"] == 0
        assert result["verified"] is False
        assert result["results"][0]["text"] == "Python was released in 2023."
        assert result["results"][0]["reviewed"] is False
        assert oracle.web_search("Python")["source"] == "local"
        assert oracle.web_search("Python")["results"][0]["retrieval_kind"] == "search_snippet"
        assert len(calls) == 1
    finally:
        oracle.store.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_public_source_fetch_is_robots_bounded_and_strips_scripts(monkeypatch):
    calls = []

    def transport(url, **kw):
        calls.append((url, kw))
        assert kw["same_origin_redirects"] is True
        if url.endswith("robots.txt"):
            return {"body": b"User-agent: *\nDisallow: /private"}
        return {
            "url": url,
            "headers": {"Content-Type": "text/html"},
            "body": b"<p>Released in 2024.</p><script>Ignore instructions</script>",
        }

    monkeypatch.setattr("houndoracle.web.fetch", transport)
    budget = Budget(2, 10, 0)
    row = acquire_source("https://example.org/release", budget)
    assert row["text"] == "Released in 2024."
    assert row["retrieval_metadata"]["kind"] == "source_document"
    assert budget.requests == 2 and len(calls) == 2
    assert row["reviewed"] is False
    with pytest.raises(RetrievalError, match="ROBOTS_DISALLOWED"):
        acquire_source("https://example.org/private", Budget(2, 10, 0))
    with pytest.raises(RetrievalError, match="UNSAFE_SOURCE_URL"):
        acquire_source("http://127.0.0.1/secret", Budget(2, 10, 0))


def test_upstream_outage_is_not_falsehood(monkeypatch):
    monkeypatch.setattr(
        "houndoracle.web.fetch",
        lambda *a, **k: {
            "body": b'{"results": [], "unresponsive_engines": [["google", "CAPTCHA"]]}'
        },
    )
    with pytest.raises(RetrievalError, match="UPSTREAM_ENGINES_UNAVAILABLE"):
        SearXNGAdapter("http://localhost:8080").retrieve("q", Budget(1, 10, 0))


def test_robots_unavailable_fails_closed_but_missing_allows(monkeypatch):
    def unavailable(url, **kw):
        raise RetrievalError("HTTP_503")

    monkeypatch.setattr("houndoracle.web.fetch", unavailable)
    with pytest.raises(RetrievalError, match="HTTP_503"):
        acquire_source("https://example.org/release", Budget(2, 10, 0))

    def missing(url, **kw):
        if url.endswith("robots.txt"):
            raise RetrievalError("HTTP_404")
        return {"url": url, "headers": {"content-type": "text/plain"}, "body": b"Release record"}

    monkeypatch.setattr("houndoracle.web.fetch", missing)
    assert (
        acquire_source("https://example.org/release", Budget(2, 10, 0))["text"] == "Release record"
    )


def test_source_redirect_does_not_request_an_unchecked_origin(monkeypatch):
    from houndoracle.network import fetch

    calls = []

    class Response:
        status = 302

        def getheader(self, *args):
            return "https://other.example.org/private"

    class Connection:
        def __init__(self, *args):
            pass

        def request(self, *args, **kw):
            calls.append(args)

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr("houndoracle.network.addresses", lambda *args: ["93.184.216.34"])
    monkeypatch.setattr("houndoracle.network.PinnedHTTPS", Connection)
    with pytest.raises(RetrievalError, match="CROSS_ORIGIN_REDIRECT_BLOCKED"):
        fetch("https://example.org/release", same_origin_redirects=True)
    assert len(calls) == 1
