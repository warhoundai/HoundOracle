"""Opt-in search providers. Retrieval scores and provider agreement never certify truth."""

import hashlib
import ipaddress
import json
import math
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .budget import BudgetExceeded
from .network import HoundSearchAdapter, RetrievalError
from .network import fetch_search_api as fetch


def source_url(url):
    if not isinstance(url, str) or len(url) > 4096:
        return None
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if (
            parsed.scheme not in ("https", "http")
            or not host
            or parsed.username
            or parsed.password
            or parsed.port not in (None, 80, 443)
        ):
            return None
        if (
            any(c in url for c in ("\r", "\n", "\x00"))
            or "." not in host
            or host.endswith((".localhost", ".local", ".internal", ".invalid"))
        ):
            return None
        try:
            if not ipaddress.ip_address(host).is_global:
                return None
        except ValueError:
            pass
        query = urlencode(
            [
                (k, v)
                for k, v in parse_qsl(parsed.query, keep_blank_values=True)
                if not k.lower().startswith("utm_") and k.lower() not in ("fbclid", "gclid")
            ]
        )
        return urlunsplit((parsed.scheme, parsed.netloc.lower(), parsed.path or "/", query, ""))
    except ValueError:
        return None


def publication_timestamp(value):
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    ):
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            return (
                parsed.timestamp()
                if parsed.tzinfo is not None and parsed.timestamp() >= 0
                else None
            )
        except ValueError:
            return None
    return None


def normalize(results, provider, limit=10):
    output = []
    urls = set()
    hashes = set()
    for row in results:
        if not isinstance(row, dict):
            continue
        url = source_url(row.get("url"))
        text = row.get("text") or row.get("content")
        if isinstance(text, dict):
            text = text.get("text")
        if not isinstance(text, str) or not text.strip() or not url:
            continue
        text = text[:20000]
        digest = hashlib.sha256(" ".join(text.split()).encode()).hexdigest()
        if url in urls or digest in hashes:
            continue
        urls.add(url)
        hashes.add(digest)
        output.append(
            {
                "url": url,
                "text": text,
                "title": str(row.get("title") or "")[:1000],
                "published_at": publication_timestamp(
                    row.get("publishedDate") or row.get("published_at")
                ),
                "observed_at": time.time(),
                "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "retrieval_provider": row.get("retrieval_provider", provider)
                if provider == "federated"
                else provider,
                "reviewed": False,
                "primary": False,
                "retrieval_metadata": row.get("retrieval_metadata", {}),
            }
        )
        if len(output) >= limit:
            break
    return output


class ExaSearchAdapter:
    name = "exa"

    def __init__(
        self, api_key, *, num_results=3, request_cost_upper_usd=0.02, include_domains=None
    ):
        if (
            not isinstance(api_key, str)
            or not api_key.strip()
            or any(c in api_key for c in ("\r", "\n"))
        ):
            raise ValueError("Exa requires a nonempty API key")
        if type(num_results) is not int or not 1 <= num_results <= 10:
            raise ValueError("Exa results must be1..10")
        if not math.isfinite(request_cost_upper_usd) or request_cost_upper_usd < 0.007:
            raise ValueError("Reservation must cover reviewed auto-search base price")
        domains = list(include_domains or [])
        if len(domains) > 20 or any(
            not source_url("https://" + d) or "/" in d or "?" in d for d in domains
        ):
            raise ValueError("Invalid public source domain filter")
        self._key = api_key.strip()
        self.limit = num_results
        self.cost_upper = request_cost_upper_usd
        self.domains = domains

    def retrieve(self, query, budget):
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 4000:
            raise ValueError("Query must contain1..4000characters")
        if time.strftime("%Y-%m-%d", time.gmtime()) > "2026-12-31":
            raise RetrievalError("EXA_PRICING_REVIEW_REQUIRED")
        budget.reserve(self.cost_upper)
        payload = {
            "query": query,
            "type": "auto",
            "numResults": self.limit,
            "contents": {"text": {"maxCharacters": 20000}},
        }
        if self.domains:
            payload["includeDomains"] = self.domains
        response = fetch(
            "https://api.exa.ai/search",
            method="POST",
            body=json.dumps(payload).encode(),
            headers={"x-api-key": self._key, "Content-Type": "application/json"},
            timeout=budget.timeout(),
            max_bytes=2_000_000,
        )
        data = json.loads(response["body"])
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise RetrievalError("INVALID_EXA_RESPONSE")
        reported = (data.get("costDollars") or {}).get("total")
        if reported is not None and (
            isinstance(reported, bool)
            or not isinstance(reported, (int, float))
            or not math.isfinite(reported)
            or reported < 0
            or reported > self.cost_upper
        ):
            raise RetrievalError("EXA_COST_BOUND_EXCEEDED_OR_INVALID")
        meta = {
            "request_id": data.get("requestId"),
            "reported_cost_usd": reported,
            "verified": False,
        }
        return normalize(
            [r | {"retrieval_metadata": meta} for r in data["results"] if isinstance(r, dict)],
            self.name,
            self.limit,
        )


class WikipediaSearchAdapter:
    """Free public encyclopedia discovery, not an independent confirmation of copied sources."""

    name = "wikipedia"

    def __init__(self, num_results=3):
        if type(num_results) is not int or not 1 <= num_results <= 10:
            raise ValueError("Results must be1..10")
        self.limit = num_results

    def retrieve(self, query, budget):
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 4000:
            raise ValueError("Invalid search query")
        budget.reserve(0)
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": self.limit,
            "prop": "extracts|info",
            "explaintext": 1,
            "exintro": 1,
            "inprop": "url",
            "format": "json",
            "formatversion": 2,
        }
        response = fetch(
            "https://en.wikipedia.org/w/api.php?" + urlencode(params),
            timeout=budget.timeout(),
            max_bytes=2_000_000,
        )
        data = json.loads(response["body"])
        if not isinstance(data, dict) or "error" in data:
            raise RetrievalError("INVALID_WIKIPEDIA_RESPONSE")
        pages = (data.get("query") or {}).get("pages", [])
        if not isinstance(pages, list):
            raise RetrievalError("INVALID_WIKIPEDIA_RESPONSE")
        return normalize(
            [
                {"url": p.get("fullurl"), "text": p.get("extract"), "title": p.get("title")}
                for p in sorted(
                    (p for p in pages if isinstance(p, dict)), key=lambda p: p.get("index", 0)
                )
            ],
            self.name,
            self.limit,
        )


class FederatedSearch:
    """Bounded fallback or fusion; provider count is never source independence."""

    name = "federated"

    def __init__(self, providers, *, mode="fallback"):
        if not providers or len(providers) > 4 or mode not in ("fallback", "fusion"):
            raise ValueError("Invalid search federation")
        self.providers = list(providers)
        self.mode = mode
        self._status = threading.local()

    @property
    def last_status(self):
        return getattr(self._status, "value", {})

    def retrieve(self, query, budget):
        results = []
        self._status.value = {}
        completed = False
        for provider in self.providers:
            try:
                hits = provider.retrieve(query, budget)
                completed = True
                self._status.value[provider.name] = getattr(provider, "last_status", {})
                results.extend(normalize(hits, provider.name))
                if results and self.mode == "fallback":
                    break
            except BudgetExceeded:
                if results:
                    return results
                raise
            except (RetrievalError, ValueError, KeyError, TypeError, OSError):
                self._status.value[provider.name] = getattr(provider, "last_status", {})
                continue
        if not completed:
            raise RetrievalError("ALL_SEARCH_PROVIDERS_FAILED")
        # Cross-provider duplicate URLs/spans collapse. Preserve first-provider provenance.
        unique = []
        seen_urls = set()
        seen_text = set()
        for row in results:
            key = hashlib.sha256(" ".join(row["text"].split()).encode()).hexdigest()
            if row["url"] not in seen_urls and key not in seen_text:
                unique.append(row)
                seen_urls.add(row["url"])
                seen_text.add(key)
        return unique[:10]


def provider_from_environment():
    """Construct only; no network call, key logging or private query submission on startup."""
    key = os.environ.get("EXA_API_KEY")
    if not key and os.environ.get("EXA_API_KEY_FILE"):
        path = Path(os.environ["EXA_API_KEY_FILE"])
        if path.stat().st_mode & 0o077:
            raise PermissionError("Exa key file must be owner-only")
        if path.stat().st_size > 4096:
            raise ValueError("Exa key file too large")
        key = path.read_text().strip()
    providers = []
    if key:
        providers.append(
            ExaSearchAdapter(
                key,
                request_cost_upper_usd=float(os.environ.get("EXA_SEARCH_COST_UPPER_USD", ".02")),
            )
        )
    if os.environ.get("HOUNDSEARCH_ENDPOINT") and os.environ.get("HOUNDSEARCH_API_KEY"):
        providers.append(
            HoundSearchAdapter(
                os.environ["HOUNDSEARCH_ENDPOINT"],
                os.environ["HOUNDSEARCH_API_KEY"],
                request_cost_upper_usd=float(os.environ["HOUNDSEARCH_COST_UPPER_USD"]),
            )
        )
    if os.environ.get("HOUNDORACLE_WIKIPEDIA_SEARCH") == "1":
        providers.append(WikipediaSearchAdapter())
    return (
        FederatedSearch(providers, mode=os.environ.get("HOUNDORACLE_SEARCH_MODE", "fallback"))
        if providers
        else None
    )
