import time

from fastapi.testclient import TestClient

from houndoracle.api import create_app
from houndoracle.jobs import Jobs
from houndoracle.models import VerifyRequest

KEY = "test-instance-key-is-at-least-24-chars"


def test_api_auth_and_semantics(oracle, ingest):
    ingest()
    with TestClient(create_app(oracle, api_key=KEY)) as client:
        assert client.get("/health").status_code == 200
        assert (
            client.post("/v1/verify", json={"question": "Was Widget released in 2024?"}).status_code
            == 401
        )
        headers = {"Authorization": "Bearer " + KEY}
        response = client.post(
            "/v1/verify", headers=headers, json={"question": "Was Widget released in 2024?"}
        )
        assert response.status_code == 200, response.text
        assert response.json()["decision"] == "YES"
        result_id = response.json()["result_id"]
        assert client.get("/v1/explanations/" + result_id, headers=headers).json()["valid"]
        assert (
            client.post("/v1/search", headers=headers, json={"query": "Widget"}).json()["verified"]
            is False
        )
        assert client.get("/v1/openapi.json", headers=headers).status_code == 200
        assert (
            client.post(
                "/v1/verify",
                headers=headers,
                json={"question": "Was Widget released in 2024?", "max_cost_usd": -1},
            ).status_code
            == 422
        )
        assert client.post("/v1/verify", headers=headers, content=b"x" * 65537).status_code == 413


def test_jobs(oracle, ingest):
    ingest()
    jobs = Jobs(oracle)
    try:
        job = jobs.submit(VerifyRequest(question="Was Widget released in 2024?"))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            result = jobs.get(job["id"])
            if result["status"] == "completed":
                break
            time.sleep(0.005)
        assert result["status"] == "completed", result
        assert result["result"]["decision"] == "YES"
    finally:
        jobs.close()


def test_mcp_registration_and_execution(oracle, ingest):
    import asyncio

    from houndoracle.mcp import server

    ingest()
    mcp = server(oracle)

    async def check():
        tools = await mcp.list_tools()
        assert {t.name for t in tools} == {"verify", "search", "explain"}
        result = await mcp.call_tool("verify", {"question": "Was Widget released in 2024?"})
        assert "YES" in str(result)

    asyncio.run(check())


def test_mcp_stdio_roundtrip(tmp_path):
    import asyncio
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def check():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "houndoracle", "--database", str(tmp_path / "mcp.sqlite"), "mcp"],
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            response = await session.call_tool(
                "verify", {"question": "Was Widget released in 2024?"}
            )
            assert not response.isError
            assert "MAYBE" in str(response.content)

    asyncio.run(check())


def test_pause_resume_cumulative_budgets(oracle):
    import threading

    started, release = threading.Event(), threading.Event()

    class Provider:
        def retrieve(self, query, budget):
            budget.reserve(0.01)
            started.set()
            release.wait(2)
            return []

    oracle.local_only = False  # Explicit legacy online-job budget test.
    oracle.provider = Provider()
    jobs = Jobs(oracle)
    try:
        job = jobs.submit(
            VerifyRequest(
                question="Was Widget released in 2024?", allow_network=True, max_cost_usd=0.01
            )
        )
        assert started.wait(1)
        jobs.transition(job["id"], "pause")
        release.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if jobs.get(job["id"])["status"] == "paused" and job["id"] not in jobs.running:
                break
            time.sleep(0.005)
        assert jobs.get(job["id"])["status"] == "paused"
        jobs.transition(job["id"], "resume")
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            result = jobs.get(job["id"])
            if result["status"] == "completed":
                break
            time.sleep(0.005)
        assert result["status"] == "completed"
        assert result["result"]["reason"] == "BUDGET_EXHAUSTED"
        assert result["result"]["usage"]["cumulative_reserved_cost_usd"] == 0.01
    finally:
        release.set()
        jobs.close()
