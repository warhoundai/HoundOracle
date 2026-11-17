"""Execute the prepared larger comparison/control loop without cloud or credentials."""

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("provider_failure", [False, True])
def test_fresh_loop_shared_preflight_records_and_call_cap(capsys, provider_failure):
    scripts = Path(__file__).parents[1] / "scripts"
    shared = (scripts / "cloud_compare_runner.py").read_text()
    source = shared[: shared.index("def main():")] + (scripts / "cloud_fresh_runner.py").read_text()
    rows = [
        {
            "id": str(i),
            "group_id": str(i),
            "question": label,
            "expected": label,
            "evidence": [] if label == "MAYBE" else [{"url": "https://example.org", "text": label}],
        }
        for i, label in enumerate(["YES", "NO", "MAYBE"])
    ]
    namespace = {"__name__": "fixture", "PAYLOAD": {"project": "fixture", "confirmation": rows}}
    exec(compile(source, "<fixture-cloud-comparison>", "exec"), namespace)  # noqa: S102 -- trusted local runner under test

    class Ledger:
        def __init__(self, *a, **k):
            pass

        def close(self):
            pass

    class Gemini:
        def __init__(self, *a, **k):
            self.failed = False

        def answer(self, question, evidence):
            if provider_failure and question == "NO" and not self.failed:
                self.failed = True
                raise TimeoutError("Fixture provider timeout")
            return {
                "answer": {"decision": question},
                "output_valid": True,
                "reported_model_version": "gemini-3.8-flash",
                "token_cost_at_list_price_usd": 0.001,
            }

    class Oracle:
        def __init__(self):
            self.artifact = {"population": "fixture"}

        def verify_evidence(self, question, evidence, **kwargs):
            return {"decision": "YES" if question == "YES" else "MAYBE", "confidence": 0.9}

    namespace.update(
        SpendLedger=Ledger, VertexGemini=Gemini, prepare_oracles=lambda: {"logistic": Oracle()}
    )
    namespace["main"]()
    logs = [json.loads(s) for s in capsys.readouterr().out.splitlines()]
    records = [r["record"] for r in logs if r["kind"] == "query"]
    assert len(records) == 18
    assert all(r["decision"] == r["expected"] for r in records if not r.get("failure"))
    assert sum(bool(r.get("failure")) for r in records) == int(provider_failure)
    if provider_failure:
        failed = next(r for r in records if r.get("failure"))
        assert failed["token_cost_usd"] is None and failed["reserved_usd"] > 0
        assert failed["decision"] == "MAYBE"
    assert sum(r["model_calls"] for r in records) == 9
    assert all(r["model_calls"] == 0 for r in records if r["expected"] == "MAYBE")
    counter = namespace["FreshCounter"](Gemini())
    counter.attempts = 288
    with pytest.raises(PermissionError):
        counter.answer("YES", [])
