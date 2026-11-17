"""Bounded HTTP retrieval with DNS pinning and redirect revalidation."""

import http.client
import ipaddress
import json
import math
import socket
import ssl
import time
from urllib.parse import urljoin, urlsplit


class RetrievalError(Exception):
    pass


def addresses(host, port):
    results = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    ips = sorted({r[4][0] for r in results})
    if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
        raise RetrievalError("PRIVATE_NETWORK_BLOCKED")
    return ips


class PinnedHTTP(http.client.HTTPConnection):
    def __init__(self, host, port, ip, timeout):
        super().__init__(host, port, timeout=timeout)
        self.ip = ip

    def connect(self):
        self.sock = socket.create_connection((self.ip, self.port), self.timeout)


class PinnedHTTPS(PinnedHTTP):
    def connect(self):
        super().connect()
        self.sock = ssl.create_default_context().wrap_socket(self.sock, server_hostname=self.host)


def fetch(
    url,
    *,
    timeout=5,
    max_bytes=1_000_000,
    headers=None,
    method="GET",
    body=None,
    allow_host=None,
    same_origin_redirects=False,
    accepted_statuses=(200, 304),
):
    deadline = time.monotonic() + timeout
    original = urlsplit(url)
    headers = dict(headers or {})
    for _ in range(4):
        parsed = urlsplit(url)
        if (
            parsed.scheme not in ("https", "http")
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise RetrievalError("UNSAFE_URL")
        if parsed.fragment or any(c in url for c in ("\r", "\n", "\x00")):
            raise RetrievalError("UNSAFE_URL")
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if port not in (80, 443) and allow_host != (host, port):
            raise RetrievalError("UNSAFE_PORT")
        # Private endpoints require an exact, operator-configured adapter origin.
        ips = (
            [r[4][0] for r in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]
            if allow_host == (host, port)
            else addresses(host, port)
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RetrievalError("TIME_EXHAUSTED")
        cls = PinnedHTTPS if parsed.scheme == "https" else PinnedHTTP
        connection = cls(host, port, ips[0], remaining)
        try:
            connection.request(
                method,
                (parsed.path or "/") + ("?" + parsed.query if parsed.query else ""),
                body=body,
                headers={"User-Agent": "HoundOracle/0.1", "Accept-Encoding": "identity", **headers},
            )
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                next_url = urljoin(url, response.getheader("Location", ""))
                target = urlsplit(next_url)
                if same_origin_redirects and (target.scheme, target.netloc) != (
                    original.scheme,
                    original.netloc,
                ):
                    raise RetrievalError("CROSS_ORIGIN_REDIRECT_BLOCKED")
                # Credentialed adapters and POSTs never follow redirects.
                if method != "GET" or any(
                    k.lower() in {"authorization", "x-api-key", "cookie", "proxy-authorization"}
                    for k in headers
                ):
                    raise RetrievalError("ADAPTER_REDIRECT_BLOCKED")
                if original.scheme == "https" and target.scheme != "https":
                    raise RetrievalError("TLS_DOWNGRADE_BLOCKED")
                url = next_url
                continue
            if response.status not in accepted_statuses:
                raise RetrievalError("HTTP_" + str(response.status))
            data = bytearray()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RetrievalError("TIME_EXHAUSTED")
                if connection.sock:
                    connection.sock.settimeout(remaining)
                chunk = response.read1(min(65536, max_bytes + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > max_bytes:
                    raise RetrievalError("DOCUMENT_TOO_LARGE")
            return {
                "url": url,
                "status": response.status,
                "headers": dict(response.getheaders()),
                "body": bytes(data),
            }
        finally:
            connection.close()
    raise RetrievalError("TOO_MANY_REDIRECTS")


class HoundSearchAdapter:
    """HTTP boundary preserves HoundSearch's independent Python 3.12 runtime.

    An explicit per-request upper cost bound must include all downstream work.
    Provider is invoked only when the verification request allows networking.
    """

    name = "houndsearch"

    def __init__(self, endpoint, api_key, *, request_cost_upper_usd, allow_private_endpoint=False):
        if not math.isfinite(request_cost_upper_usd) or request_cost_upper_usd <= 0:
            raise ValueError("HoundSearch requires a positive downstream spending cap")
        self.endpoint = endpoint.rstrip("/")
        parsed = urlsplit(endpoint)
        if parsed.scheme != "https" and not (
            allow_private_endpoint
            and parsed.scheme == "http"
            and parsed.hostname in ("localhost", "127.0.0.1", "::1")
        ):
            raise ValueError("Remote adapters require HTTPS")
        self.api_key = api_key
        self.cost_upper = request_cost_upper_usd
        self.allow_host = (
            (parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
            if allow_private_endpoint
            else None
        )

    def retrieve(self, query, budget):
        budget.reserve(self.cost_upper)
        payload = {
            "query": query,
            "include_contents": True,
            "limit": 2,
            "budget": {
                "max_requests": 4,
                "max_steps": 1,
                "max_cost_usd": self.cost_upper,
            },
        }
        # The downstream service must independently enforce its configured spend cap.
        result = fetch(
            self.endpoint + "/v1/search",
            method="POST",
            body=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
            timeout=budget.timeout(),
            allow_host=self.allow_host,
        )
        return json.loads(result["body"])["results"]


def fetch_search_api(
    url, *, timeout=10, max_bytes=2_000_000, headers=None, method="GET", body=None
):
    """Proxy-compatible transport ONLY for fixed public search-service origins.

    Arbitrary source documents still use DNS-pinned fetch(). No redirects, even
    within the same host; credentials are never forwarded to another origin.
    """
    import urllib.error
    import urllib.request

    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"api.exa.ai", "en.wikipedia.org"}
        or parsed.port not in (None, 443)
        or parsed.username
        or parsed.password
        or parsed.fragment
        or any(c in url for c in ("\r", "\n", "\x00"))
    ):
        raise RetrievalError("UNSAFE_SEARCH_ENDPOINT")
    if (parsed.hostname, parsed.path, method) not in {
        ("api.exa.ai", "/search", "POST"),
        ("en.wikipedia.org", "/w/api.php", "GET"),
    }:
        raise RetrievalError("UNSAFE_SEARCH_ENDPOINT")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            raise RetrievalError("ADAPTER_REDIRECT_BLOCKED")

    opener = urllib.request.build_opener(NoRedirect())
    deadline = time.monotonic() + timeout
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "User-Agent": "HoundOracle/0.1 (evidence discovery)",
            "Accept-Encoding": "identity",
            **(headers or {}),
        },
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            output = bytearray()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RetrievalError("TIME_EXHAUSTED")
                sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
                if sock:
                    sock.settimeout(remaining)
                chunk = response.read1(min(65536, max_bytes + 1 - len(output)))
                if not chunk:
                    break
                output.extend(chunk)
                if len(output) > max_bytes:
                    raise RetrievalError("DOCUMENT_TOO_LARGE")
            return {
                "url": url,
                "status": response.status,
                "headers": dict(response.headers),
                "body": bytes(output),
            }
    except urllib.error.HTTPError as exc:
        exc.close()
        raise RetrievalError("HTTP_" + str(exc.code)) from None
    except (urllib.error.URLError, OSError, TimeoutError):
        raise RetrievalError("SEARCH_TRANSPORT_FAILED") from None
