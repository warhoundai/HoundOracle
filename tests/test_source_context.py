"""Document context is scoped metadata and cannot change the default evidence path."""

from copy import deepcopy

from houndoracle.experiments.source_context import (
    SourceContextNLI,
    document_title,
    grounding_features,
)


def test_context_copies_input_and_distinguishes_same_named_documents():
    class Model:
        def __init__(self):
            self.config = {"revision": "fixture"}

        def predict(self, question, evidence):
            self.received = question, evidence
            return {"candidate": "MAYBE"}

    model = Model()
    wrapper = SourceContextNLI(model)
    evidence = [
        {"page": "Love_%26_Friendship", "text": "It stars an actress."},
        {"page": "Love_At_First_Fight", "text": "It stars an actress."},
    ]
    original = deepcopy(evidence)
    assert wrapper.predict("Who stars?", evidence)["candidate"] == "MAYBE"
    assert evidence == original
    passages = model.received[1]
    assert passages[0]["text"] != passages[1]["text"]
    assert "Love & Friendship" in passages[0]["text"]
    assert wrapper.config != model.config


def test_missing_metadata_is_not_invented_or_treated_as_entity_proof():
    assert document_title({"url": "https://example.org/Claim"}) == ""
    assert document_title({"page": None}) == ""
    assert document_title({"page": "a" * 1000}) == "a" * 300
    assert grounding_features("Claim", []) == [0, 0, 0, 0]
    assert grounding_features("Love & Friendship", [{"page": "Love_At_First_Fight"}])[-1] == 0


def test_title_cannot_manufacture_evidence_when_passage_is_blank():
    class Model:
        def __init__(self):
            self.config = {}

        def predict(self, question, evidence):
            assert [item["text"] for item in evidence] == ["", "   "]
            return {"candidate": "MAYBE"}

    result = SourceContextNLI(Model()).predict(
        "A claim", [{"page": "A_claim", "text": ""}, {"page": "A_claim", "text": "   "}]
    )
    assert result["candidate"] == "MAYBE"
