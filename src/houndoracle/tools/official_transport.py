"""Operator-opted environment proxy for a fixed official-source allowlist.

Questions cannot supply URLs. Redirects are disabled and direct DNS is checked;
when enabled, the configured proxy remains a trusted network dependency.
"""

import re
import time
import urllib.error
import urllib.request

from ..network import RetrievalError, addresses

ALLOWED = re.compile(
    r"https://nodejs\.org/(?:dist/index\.json|en/blog/release/v\d{1,3}\.\d{1,3}\.\d{1,3})\Z"
)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RetrievalError("OFFICIAL_SOURCE_REDIRECT_BLOCKED")


def fetch_official(url, *, timeout=10, max_bytes=2_000_000, **options):
    if not isinstance(url, str) or not ALLOWED.fullmatch(url):
        raise RetrievalError("OFFICIAL_SOURCE_NOT_ALLOWED")
    if not 0 < timeout <= 10 or not 1 <= max_bytes <= 2_000_000:
        raise ValueError("Transport envelope exceeded")
    addresses("nodejs.org", 443)
    start = time.monotonic()
    opener = urllib.request.build_opener(NoRedirect())
    request = urllib.request.Request(
        url, headers={"User-Agent": "HoundOracle/0.1", "Accept-Encoding": "identity"}
    )
    with opener.open(request, timeout=timeout) as response:
        if response.status != 200 or response.url != url:
            raise RetrievalError("UNEXPECTED_OFFICIAL_RESPONSE")
        body = bytearray()
        while True:
            if time.monotonic() - start > timeout:
                raise RetrievalError("TIME_EXHAUSTED")
            chunk = response.read1(min(65536, max_bytes + 1 - len(body)))
            if not chunk:
                break
            body.extend(chunk)
            if len(body) > max_bytes:
                raise RetrievalError("DOCUMENT_TOO_LARGE")
        return {"url": url, "status": 200, "body": bytes(body), "headers": dict(response.headers)}
