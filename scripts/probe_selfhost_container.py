"""Run inside a network=none container; use loopback for its authenticated local API."""

import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=0.2)
    except OSError:
        isolated = True
    else:
        raise AssertionError("Container unexpectedly has outbound networking")
    base = "http://127.0.0.1:8083"
    for _ in range(60):
        try:
            with urllib.request.urlopen(base + "/health", timeout=1) as r:
                assert r.status == 200
            break
        except OSError:
            time.sleep(0.2)
    else:
        raise AssertionError("Selfhosted API did not start")

    def post(path, payload):
        request = urllib.request.Request(
            base + path,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": "Bearer " + "offline-test-access-key-not-a-cloud-key",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.load(response)

    search = post("/v1/search", {"query": "Widget"})
    assert len(search["results"]) == 1 and search["verified"] is False
    verdict = post(
        "/v1/verify",
        {"question": "Was Widget released in 2024?", "allow_network": True, "max_cost_usd": 1},
    )
    assert verdict["decision"] == "MAYBE" and verdict["usage"]["requests"] == 0
    try:
        post(
            "/v1/search", {"query": "UnfindableZXQ90210", "allow_network": True, "max_cost_usd": 1}
        )
    except urllib.error.HTTPError as exc:
        assert exc.code == 403
    else:
        raise AssertionError("Offline server allowed an external search")
    result = {
        "container_network_isolated": isolated,
        "authenticated_api_search_passed": True,
        "unsupported_or_uncalibrated_verification_abstains": True,
        "hosted_search_requests": 0,
        "hosted_model_requests": 0,
        "remote_search_rejected": True,
    }
    Path("/tmp/container-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
