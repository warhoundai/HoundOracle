import base64
import json
import subprocess

import pytest

from houndoracle.captcha import solve_encoded_captcha, solve_text_captcha


def test_unsupported_interactive_challenge_never_starts_backend(monkeypatch):
    monkeypatch.setattr(
        "houndoracle.captcha.subprocess.Popen", lambda *a, **k: pytest.fail("unexpected solver")
    )
    result = solve_text_captcha(b"image", challenge_type="recaptcha")
    assert result["status"] == "UNSUPPORTED_CHALLENGE" and result["server_accepted"] is False


def test_input_and_alphabet_bounds():
    for data in [b"", b"a" * 1_000_001, "https://example.org/captcha"]:
        with pytest.raises(ValueError):
            solve_text_captcha(data)
    with pytest.raises(ValueError):
        solve_encoded_captcha("not base64!")
    with pytest.raises(ValueError):
        solve_text_captcha(b"img", alphabet="ABC;rm -rf /")
    with pytest.raises(ValueError):
        solve_text_captcha(b"img", max_seconds=float("inf"))


def test_worker_does_not_receive_credentials_or_claim_server_acceptance(monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "private-key")

    class Process:
        returncode = 0

        def __init__(self, cmd, **kw):
            assert (
                cmd[-1] == "houndoracle.captcha_worker" and "AWS_SECRET_ACCESS_KEY" not in kw["env"]
            )
            assert kw["env"]["OMP_NUM_THREADS"] == "1" and "shell" not in kw

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def communicate(self, payload, **kw):
            assert base64.b64decode(json.loads(payload)["image"]) == b"image"
            return b'{"status":"CANDIDATE","candidate":"AB12","backend_version":"fixture"}', b""

    monkeypatch.setattr("houndoracle.captcha.subprocess.Popen", Process)
    result = solve_text_captcha(b"image", expected_length=4)
    assert result["candidate"] == "AB12" and not result["server_accepted"]
    assert solve_text_captcha(b"image", expected_length=5)["status"] == "LENGTH_MISMATCH"


def test_timeout_terminates_worker_process_group(monkeypatch):
    import os

    killed = []

    class Process:
        pid = 98765
        returncode = -9

        def __init__(self, *a, **kw):
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def communicate(self, *a, **kw):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("worker", 1)
            return b"", b""

        def kill(self):
            killed.append(self.pid)

    monkeypatch.setattr("houndoracle.captcha.subprocess.Popen", Process)
    if os.name == "posix":
        monkeypatch.setattr("houndoracle.captcha.os.killpg", lambda pid, sig: killed.append(pid))
    result = solve_text_captcha(b"image", max_seconds=1)
    assert result["status"] == "TIME_BUDGET_EXCEEDED" and killed == [98765]


def test_captcha_rest_tool_is_opt_in_and_authenticated(monkeypatch):
    from fastapi.testclient import TestClient

    from houndoracle import Oracle
    from houndoracle.api import create_app

    oracle = Oracle()
    key = "local-test-access-key-at-least-24"
    with TestClient(create_app(oracle, api_key=key)) as client:
        assert (
            client.post(
                "/v1/captcha/solve",
                headers={"Authorization": "Bearer " + key},
                json={"image_base64": "aW1n"},
            ).status_code
            == 404
        )
    monkeypatch.setattr(
        "houndoracle.captcha.solve_encoded_captcha",
        lambda *a, **kw: {"status": "CANDIDATE", "server_accepted": False},
    )
    with TestClient(create_app(oracle, api_key=key, enable_captcha=True)) as client:
        assert client.post("/v1/captcha/solve", json={"image_base64": "aW1n"}).status_code == 401
        result = client.post(
            "/v1/captcha/solve",
            headers={"Authorization": "Bearer " + key},
            json={"image_base64": "aW1n"},
        )
        assert result.status_code == 200 and result.json()["server_accepted"] is False
    oracle.store.close()


def test_captcha_mcp_registration_and_unsupported_type(monkeypatch):
    import asyncio

    from houndoracle import Oracle
    from houndoracle.mcp import server

    oracle = Oracle()

    async def check():
        tools = await server(oracle, enable_captcha=True).list_tools()
        assert "captcha_solve" in {t.name for t in tools}
        result = await server(oracle, enable_captcha=True).call_tool(
            "captcha_solve", {"image_base64": "aW1n", "challenge_type": "recaptcha"}
        )
        assert "UNSUPPORTED_CHALLENGE" in str(result)

    asyncio.run(check())
    oracle.store.close()


def test_malformed_image_is_rejected_by_actual_worker():
    # Optional OCR packages are unnecessary for the core suite.
    result = solve_text_captcha(b"not an image", max_seconds=5)
    assert result["status"] in {"INVALID_IMAGE", "BACKEND_UNAVAILABLE"}
    assert result["candidate"] is None and result["server_accepted"] is False
