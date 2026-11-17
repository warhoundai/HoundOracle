"""Read-only HTTP integrations with independently deployed WarHound public tools."""

import json
import os
import re
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from .network import RetrievalError, addresses, fetch
from .search import normalize, source_url


def challenge_detected(text):
    # Conservative quarantine, not evidence classification or a solver.
    sample = text[:20000].lower()
    return any(
        s in sample
        for s in (
            "verify you are human",
            "verify that you are human",
            "unusual traffic from your",
            "complete the captcha",
            "g-recaptcha",
            "h-captcha",
            "cf-chl-",
        )
    )


class LocalService:
    def __init__(self, endpoint, key):
        p = urlsplit(endpoint)
        if (
            p.scheme not in {"http", "https"}
            or not p.hostname
            or p.username
            or p.password
            or p.path not in ("", "/")
            or p.query
            or p.fragment
            or any(c in endpoint for c in ("\r", "\n", "\x00"))
        ):
            raise ValueError("Service requires an operator-configured origin")
        if p.scheme == "http" and p.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Use HTTPS or loopback HTTP for WarHound services")
        if not isinstance(key, str) or not key or any(c in key for c in ("\r", "\n")):
            raise ValueError("A local service access key is required")
        self.endpoint, self._key = endpoint.rstrip("/"), key
        self.allow_host = (p.hostname, p.port or (443 if p.scheme == "https" else 80))

    def request(self, path, budget, payload=None, method=None, params=None, idempotency_key=None):
        if not path.startswith("/v1/") or any(c in path for c in ("?", "#", "..")):
            raise ValueError("Unsupported service path")
        budget.reserve(0)
        response = fetch(
            self.endpoint + path + ("?" + urlencode(params) if params else ""),
            method=method or ("POST" if payload is not None else "GET"),
            body=json.dumps(payload).encode() if payload is not None else None,
            headers={
                "Authorization": "Bearer " + self._key,
                "Content-Type": "application/json",
                **({"Idempotency-Key": idempotency_key} if idempotency_key else {}),
            },
            timeout=budget.timeout(),
            max_bytes=2_000_000,
            allow_host=self.allow_host,
            accepted_statuses=(200, 201, 202, 204),
        )
        if not response["body"]:
            return {}
        data = json.loads(response["body"])
        if not isinstance(data, dict):
            raise RetrievalError("INVALID_WARHOUND_RESPONSE")
        return data


class HoundSourceAdapter(LocalService):
    def __init__(self, endpoint, key, *, tool="houndcrawler", render="never"):
        super().__init__(endpoint, key)
        if tool not in {"houndcrawler", "houndscraper"} or render not in {
            "never",
            "auto",
            "always",
        }:
            raise ValueError("Unsupported source tool or render policy")
        self.name, self.render = tool, render

    def __call__(self, url, budget):
        if not source_url(url):
            raise RetrievalError("UNSAFE_SOURCE_URL")
        addresses(
            urlsplit(url).hostname, urlsplit(url).port or (443 if url.startswith("https:") else 80)
        )
        timeout = max(1, min(15, budget.timeout()))
        result = self.request(
            "/v1/scrape",
            budget,
            {
                "url": url,
                "options": {
                    "formats": ["text"],
                    "render": self.render,
                    "respect_robots": True,
                    "timeout_seconds": timeout,
                    "browser_timeout_seconds": timeout,
                    "max_bytes": 1_000_000,
                    "max_age": 300,
                },
            },
        )
        target = source_url(result.get("final_url", result.get("url")))
        if not target or urlsplit(target).netloc != urlsplit(url).netloc:
            raise RetrievalError("UNSAFE_SOURCE_RESPONSE_URL")
        text = (result.get("data") or {}).get("text")
        if not isinstance(text, str) or not text.strip():
            raise RetrievalError("SOURCE_TEXT_UNAVAILABLE")
        if challenge_detected(text):
            raise RetrievalError("CAPTCHA_REQUIRES_INTERACTION")
        return normalize(
            [
                {
                    "url": target,
                    "text": text,
                    "retrieval_metadata": {
                        "kind": "source_document",
                        "acquisition": result.get("acquisition"),
                    },
                }
            ],
            self.name,
            1,
        )[0]

    def submit_crawl(self, url, budget, *, max_pages=20):
        if self.name != "houndcrawler":
            raise ValueError(
                "Bounded multi-page jobs require HoundCrawler; use HoundScraper for single pages"
            )
        if not source_url(url) or type(max_pages) is not int or not 1 <= max_pages <= 20:
            raise ValueError("Public URL and max_pages 1..20 required")
        addresses(urlsplit(url).hostname, 443 if url.startswith("https:") else 80)
        return self.request(
            "/v1/jobs",
            budget,
            {
                "kind": "crawl",
                "urls": [url],
                "max_pages": max_pages,
                "max_depth": 2,
                "max_concurrency": 1,
                "max_seconds": 60,
                "max_download_bytes": 5_000_000,
                "allow_external_links": False,
                "allow_subdomains": False,
                "sitemap": "skip",
                "options": {
                    "formats": ["text"],
                    "render": "never",
                    "respect_robots": True,
                    "max_bytes": 1_000_000,
                },
            },
            idempotency_key=uuid.uuid4().hex,
        )

    def crawl_results(self, job_id, budget, *, cursor=0):
        if not re.fullmatch(r"[0-9a-f]{32}", job_id) or type(cursor) is not int or cursor < 0:
            raise ValueError("Invalid crawl job ID or cursor")
        page = self.request(
            "/v1/jobs/" + job_id + "/results", budget, params={"cursor": cursor, "limit": 20}
        )
        hits = []
        for row in page.get("data", [])[:20]:
            result = row.get("result") or {}
            text = (result.get("data") or {}).get("text")
            if not isinstance(text, str) or challenge_detected(text):
                continue
            hits.extend(
                normalize(
                    [
                        {
                            "url": result.get("final_url", result.get("url")),
                            "text": text,
                            "retrieval_metadata": {"kind": "source_document"},
                        }
                    ],
                    self.name,
                    1,
                )
            )
        return {"results": hits, "next_cursor": page.get("next_cursor"), "verified": False}


class HoundIndexAdapter(LocalService):
    name = "houndsearch"

    def retrieve(self, query, budget):
        # One downstream reservation is consumed by the local lookup. Any configured
        # primary upstream must reserve a second call and therefore fails closed.
        data = self.request(
            "/v1/search",
            budget,
            {
                "query": query,
                "mode": "fast",
                "limit": 3,
                "include_contents": False,
                "providers": [],
                "budget": {"max_requests": 1, "max_steps": 1},
            },
        )
        rows = []
        for row in data.get("results", [])[:3]:
            evidence = row.get("evidence") or []
            text = evidence[0].get("text") if evidence else row.get("snippet")
            rows.append(
                {
                    "url": row.get("url"),
                    "text": text,
                    "title": row.get("title"),
                    "retrieval_metadata": {
                        "kind": "source_document" if evidence else "search_snippet"
                    },
                }
            )
        return normalize(rows, self.name, 3)


class HoundBrowserAdapter(LocalService):
    """Scoped observation only; sessions always close, uncertain commands never replay."""

    def observe(self, url, budget):
        from .web import check_source_policy

        url = check_source_policy(url, budget)
        p = urlsplit(url)
        if p.scheme != "https":
            raise ValueError("HoundBrowser requires a public HTTPS origin")
        # Reserve cleanup before creating any session; even a tiny caller cap is safe.
        budget.reserve(0)
        cleanup_budget = _CleanupBudget(budget.deadline)
        session = self.request(
            "/v1/browser/sessions",
            budget,
            {"scope": {"origin": f"{p.scheme}://{p.netloc}", "methods": ["GET", "HEAD"]}},
        )
        sid = session.get("id")
        if not isinstance(sid, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", sid):
            raise RetrievalError("INVALID_BROWSER_SESSION")
        path = "/v1/browser/sessions/" + sid
        try:
            cid = uuid.uuid4().hex
            self.request(
                path + "/commands",
                budget,
                {
                    "id": cid,
                    "idempotencyKey": cid,
                    "action": "navigate",
                    "args": {"url": url},
                    "revision": session.get("revision", 0),
                },
            )
            while True:
                result = self.request(path + "/commands/" + cid, budget)
                if result.get("status") == "completed":
                    observation = result.get("result") or {}
                    text = observation.get("text", "")
                    return {
                        "status": "CAPTCHA_REQUIRES_INTERACTION"
                        if challenge_detected(text)
                        else "OBSERVED",
                        "observation": observation,
                        "verified": False,
                    }
                if result.get("status") in {"failed", "cancelled", "outcome_unknown"}:
                    raise RetrievalError("BROWSER_" + result["status"].upper())
                budget.check_time()
                time.sleep(min(0.1, budget.timeout()))
        finally:
            self.request(path, cleanup_budget, method="DELETE")


class _CleanupBudget:
    def __init__(self, deadline):
        self.deadline = deadline

    def reserve(self, cost):
        pass  # Already reserved against caller's request cap.

    def timeout(self):
        return max(0.1, min(2, self.deadline - time.monotonic()))


def service_key(name):
    path = os.environ.get(name + "_API_KEY_FILE")
    if path:
        p = Path(path)
        if p.stat().st_mode & 0o077 or p.stat().st_size > 4096:
            raise PermissionError("Local service key file must be owner-only and bounded")
        return p.read_text().strip()
    return os.environ.get(name + "_API_KEY", "")


def source_from_environment():
    for prefix, tool in [("HOUNDCRAWLER", "houndcrawler"), ("HOUNDSCRAPER", "houndscraper")]:
        endpoint = os.environ.get(prefix + "_ENDPOINT")
        if endpoint:
            return HoundSourceAdapter(
                endpoint,
                service_key(prefix),
                tool=tool,
                render=os.environ.get("HOUNDORACLE_RENDER", "never"),
            )
    return None
