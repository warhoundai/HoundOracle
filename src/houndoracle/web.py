"""Self-hosted web discovery and bounded public-source acquisition; no model service."""

import json
import threading
import time
from html.parser import HTMLParser
from urllib.parse import urlencode, urlsplit
from urllib.robotparser import RobotFileParser

from .budget import BudgetExceeded
from .network import RetrievalError, fetch
from .search import normalize, source_url


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "template", "svg"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "template", "svg"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def plain_text(body, content_type):
    text = body.decode("utf-8", errors="replace")
    if "html" in content_type:
        parser = VisibleText()
        parser.feed(text)
        text = "\n".join(parser.parts)
    elif not content_type.startswith("text/plain"):
        raise RetrievalError("UNSUPPORTED_SOURCE_CONTENT_TYPE")
    return text[:20000]


def check_source_policy(url, budget):
    """Validate a public source and its robots policy before acquisition."""
    url = source_url(url)
    if not url:
        raise RetrievalError("UNSAFE_SOURCE_URL")
    parsed = urlsplit(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    budget.reserve(0)
    try:
        response = fetch(
            origin + "/robots.txt",
            timeout=budget.timeout(),
            max_bytes=100_000,
            same_origin_redirects=True,
        )
    except RetrievalError as exc:
        if str(exc) not in {"HTTP_404", "HTTP_410"}:
            raise
        response = {"body": b""}
    rules = RobotFileParser()
    rules.parse(response["body"].decode("utf-8", errors="replace").splitlines())
    if not rules.can_fetch("HoundOracle", url):
        raise RetrievalError("ROBOTS_DISALLOWED")
    # Never sleep through a request deadline or silently ignore a site rate policy.
    if rules.crawl_delay("HoundOracle") or rules.request_rate("HoundOracle"):
        raise RetrievalError("ROBOTS_RATE_POLICY_REQUIRES_SCHEDULED_CRAWLER")
    return url


def acquire_source(url, budget):
    """Bounded public source fetch; no link/script execution or unchecked redirects."""
    url = check_source_policy(url, budget)
    parsed = urlsplit(url)
    budget.reserve(0)
    response = fetch(url, timeout=budget.timeout(), max_bytes=1_000_000, same_origin_redirects=True)
    headers = {k.lower(): v for k, v in response["headers"].items()}
    final = source_url(response["url"])
    if not final or urlsplit(final).netloc != parsed.netloc:
        raise RetrievalError("CROSS_ORIGIN_SOURCE_REDIRECT_REQUIRES_ROBOTS_RECHECK")
    from .warhound import challenge_detected

    text = plain_text(response["body"], headers.get("content-type", ""))
    if challenge_detected(text):
        raise RetrievalError("CAPTCHA_REQUIRES_INTERACTION")
    return (
        normalize(
            [{"url": final, "text": text, "retrieval_metadata": {"kind": "source_document"}}],
            "local-crawler",
        )[0]
        if text.strip()
        else None
    )


class SearXNGAdapter:
    """An operator-configured SearXNG server. Google remains an upstream engine.

    JSON must be enabled on that server. Engine failures are not false claims.
    Engine-internal requests are bounded by SearXNG settings, not our HTTP counter.
    """

    name = "searxng"

    def __init__(
        self,
        endpoint,
        *,
        engines="google,bing,duckduckgo",
        fetch_sources=True,
        source_acquirer=None,
    ):
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
            or any(c in endpoint for c in ("\r", "\n", "\x00"))
        ):
            raise ValueError("SearXNG requires an operator-configured origin")
        if parsed.scheme == "http" and parsed.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
            "searxng",
        }:
            raise ValueError("Use HTTPS except for a local SearXNG service")
        selected = engines.split(",")
        if (
            not selected
            or len(selected) > 3
            or any(e not in {"google", "bing", "duckduckgo"} for e in selected)
        ):
            raise ValueError("Choose up to three supported web engines")
        self.endpoint = endpoint.rstrip("/")
        self.engines = engines
        self.fetch_sources = fetch_sources
        self.source_acquirer = source_acquirer or acquire_source
        self._cooldowns = {}
        self._lock = threading.Lock()
        self._status = threading.local()
        self.allow_host = (
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
        )

    @property
    def last_status(self):
        return getattr(self._status, "value", {})

    def retrieve(self, query, budget):
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 4000:
            raise ValueError("Invalid search query")
        now = time.monotonic()
        with self._lock:
            active = [e for e in self.engines.split(",") if self._cooldowns.get(e, 0) <= now]
        self._status.value = {"active_engines": active, "failures": []}
        if not active:
            raise RetrievalError("SEARCH_ENGINES_COOLDOWN")
        budget.reserve(0)
        response = fetch(
            self.endpoint
            + "/search?"
            + urlencode(
                {
                    "q": query,
                    "format": "json",
                    "engines": ",".join(active),
                    "language": "en-US",
                    "categories": "general",
                    "safesearch": 1,
                }
            ),
            timeout=budget.timeout(),
            max_bytes=2_000_000,
            allow_host=self.allow_host,
        )
        data = json.loads(response["body"])
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise RetrievalError("INVALID_SEARXNG_RESPONSE")
        failures = []
        for row in data.get("unresponsive_engines", [])[:10]:
            if not isinstance(row, (list, tuple)) or len(row) != 2 or row[0] not in active:
                continue
            message = str(row[1]).lower()
            code = (
                "CAPTCHA_REQUIRED"
                if "captcha" in message
                else (
                    "ACCESS_DENIED"
                    if "denied" in message
                    else (
                        "RATE_LIMITED"
                        if "429" in message or "many requests" in message
                        else "UPSTREAM_FAILED"
                    )
                )
            )
            failures.append({"engine": row[0], "code": code})
            with self._lock:
                self._cooldowns[row[0]] = now + (600 if code == "CAPTCHA_REQUIRED" else 180)
        self._status.value = {"active_engines": active, "failures": failures}
        rows = []
        for row in data["results"][:30]:
            if not isinstance(row, dict):
                continue
            snippet = VisibleText()
            snippet.feed(str(row.get("content") or ""))
            rows.append(
                {
                    "url": row.get("url"),
                    "text": " ".join(snippet.parts),
                    "title": row.get("title"),
                    "retrieval_metadata": {
                        "kind": "search_snippet",
                        "engines": row.get("engines", []),
                    },
                }
            )
        hits = normalize(rows, self.name, 3)
        if not hits and data.get("unresponsive_engines"):
            raise RetrievalError("SEARXNG_UPSTREAM_ENGINES_UNAVAILABLE")
        if self.fetch_sources:
            # One page, at most two additional requests; no unbounded spidering.
            for index, hit in enumerate(hits):
                try:
                    acquired = self.source_acquirer(hit["url"], budget)
                    if acquired:
                        acquired["title"] = hit["title"]
                        hits[index] = acquired
                    break
                except BudgetExceeded:
                    break
                except (RetrievalError, OSError, ValueError, IndexError):
                    continue
        return hits
