import json

import pytest
from test_experiments import fake_client

from houndoracle.experiments.citations import citations_match
from houndoracle.experiments.vertex import SpendLedger, VertexConfig, VertexGemini


@pytest.mark.parametrize(
    "citations",
    [
        [],
        None,
        [{"url": "https://evil.example", "quote": "Fact."}],
        [{"url": "https://example.org", "quote": "Invented fact."}],
        [{"url": "https://example.org", "quote": " "}],
        ["https://example.org"],
    ],
)
def test_definitive_uncited_or_altered_outputs_fail_closed(tmp_path, citations):
    client = fake_client()
    original = client.models.generate_content

    def generate(**kwargs):
        response = original(**kwargs)
        response.text = json.dumps({"decision": "YES", "citations": citations})
        return response

    client.models.generate_content = generate
    ledger = SpendLedger(tmp_path / "spend.sqlite", authorized_usd=1)
    result = VertexGemini(VertexConfig(project="fixture"), ledger, client=client).answer(
        "Claim", [{"url": "https://example.org", "text": "Fact."}]
    )
    assert result["answer"]["decision"] == "MAYBE"
    assert not result["output_valid"] and result["failure_reason"] == "UNVERIFIABLE_CITATION"
    assert result["token_cost_at_list_price_usd"] > 0  # Failed output still costs money.
    ledger.close()


def test_exact_quote_integrity_does_not_imply_entailment():
    evidence = [{"url": "https://example.org", "text": "Contradictory fact."}]
    answer = {
        "decision": "YES",
        "citations": [{"URL": "https://example.org", "quote": "Contradictory fact."}],
    }
    assert citations_match(answer, evidence)  # Semantic support remains a separate gate.
    answer["citations"].append({"url": "https://unknown.example", "quote": "Contradictory fact."})
    assert not citations_match(answer, evidence)
