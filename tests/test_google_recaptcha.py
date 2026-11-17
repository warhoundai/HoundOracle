import json

import pytest

from houndoracle.recaptcha import solve_google_recaptcha


def test_permission_and_target_validation_before_process(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "houndoracle.recaptcha.subprocess.Popen", lambda *a, **k: pytest.fail("unexpected process")
    )
    options = {"helper_directory": tmp_path, "model_directory": tmp_path}
    with pytest.raises(PermissionError):
        solve_google_recaptcha("https://www.google.com/", **options)
    for url in [
        "http://169.254.169.254/",
        "https://www.google.com.evil.example/",
        "https://user@www.google.com/",
        "https://www.google.com/#token",
    ]:
        with pytest.raises(ValueError):
            solve_google_recaptcha(url, allow_network=True, **options)


def test_process_boundary_strips_credentials_and_solver_secrets(tmp_path, monkeypatch):
    (tmp_path / "bridge.mjs").write_text("// fixture")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "private-cloud-key")

    class Process:
        returncode = 0

        def __init__(self, cmd, **kw):
            assert cmd[0] == "node" and "shell" not in kw
            assert "AWS_SECRET_ACCESS_KEY" not in kw["env"]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def communicate(self, payload, **kw):
            body = json.loads(payload)
            assert (
                body["port"] == 9222
                and body["target_url"] == "https://www.google.com/recaptcha/api2/demo"
            )
            return json.dumps(
                {
                    "status": "WIDGET_SOLVED",
                    "widget_solved": True,
                    "verification": "widget_and_token_confirmed",
                    "token": "private-response-token",
                    "cookies": [{"value": "private-cookie"}],
                }
            ).encode(), b""

    monkeypatch.setattr("houndoracle.recaptcha.subprocess.Popen", Process)
    result = solve_google_recaptcha(
        "https://www.google.com/recaptcha/api2/demo",
        helper_directory=tmp_path,
        model_directory=tmp_path,
        allow_network=True,
    )
    assert result["widget_solved"] is True and result["server_accepted"] is None
    assert "private" not in json.dumps(result)


def test_google_tool_is_opt_in_and_operator_controls_browser_config(tmp_path, monkeypatch):
    import asyncio

    from fastapi.testclient import TestClient

    from houndoracle import Oracle
    from houndoracle.api import create_app
    from houndoracle.mcp import server

    calls = []

    def solver(url, **options):
        calls.append((url, options))
        return {"status": "UNVERIFIED", "widget_solved": False, "server_accepted": None}

    monkeypatch.setattr("houndoracle.recaptcha.solve_google_recaptcha", solver)
    config = {"helper_directory": tmp_path, "model_directory": tmp_path, "port": 9222}
    oracle = Oracle()
    key = "local-google-test-service-access-key"
    with TestClient(create_app(oracle, api_key=key, recaptcha_config=config)) as client:
        assert (
            client.post(
                "/v1/captcha/google", json={"target_url": "https://www.google.com/"}
            ).status_code
            == 401
        )
        result = client.post(
            "/v1/captcha/google",
            headers={"Authorization": "Bearer " + key},
            json={
                "target_url": "https://www.google.com/recaptcha/api2/demo",
                "allow_network": True,
            },
        )
        assert result.status_code == 200 and not result.json()["widget_solved"]

    async def check():
        tools = await server(oracle, recaptcha_config=config).list_tools()
        assert "recaptcha_solve" in {t.name for t in tools}

    asyncio.run(check())
    assert calls[0][1]["port"] == 9222 and calls[0][1]["allow_network"] is True
    oracle.store.close()
