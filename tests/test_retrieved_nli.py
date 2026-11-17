from houndoracle.experiments.retrieved_nli import RetrievedNLI


def test_context_shortcut_cannot_qualify_unknown_or_multihop_claims():
    class Model:
        def __init__(self):
            self.config = {}

        def predict(self, *args):
            raise AssertionError("Unsupported scope must not infer")

    result = RetrievedNLI(None, Model()).predict("A and B are linked", claim_scope="multi_hop")
    assert result["candidate"] == "MAYBE" and not result["verdict_qualified"]


def test_research_candidate_preserves_other_evidence_without_claiming_confidence():
    class Store:
        def search(self, *args):
            return [
                {"id": "primary", "data": {"url": "https://example.org/a", "text": "A"}},
                {"id": "contradiction", "data": {"url": "https://example.org/b", "text": "B"}},
            ]

    class Model:
        def __init__(self):
            self.config = {}

        def predict(self, q, evidence):
            assert [e["text"] for e in evidence] == ["A"]
            return {"candidate": "YES", "raw_score": 0.999}

    result = RetrievedNLI(Store(), Model()).predict("A", claim_scope="single_document")
    assert result["remaining_evidence_document_ids"] == ["contradiction"]
    assert result["calibrated_correctness"] is None and not result["verdict_qualified"]
