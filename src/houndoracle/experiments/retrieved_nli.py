"""Research-only context selection. Never bypasses production verdict qualification."""


class RetrievedNLI:
    def __init__(self, store, model, *, policy="single-document-top1-v1"):
        if policy != "single-document-top1-v1":
            raise ValueError("Unsupported experimental retrieval policy")
        self.store, self.model = store, model
        self.config = {**model.config, "retrieval_policy": policy, "production_qualified": False}

    def predict(self, question, *, claim_scope="unknown"):
        # Narrow observed scope must be declared; do not discard sources for general
        # multi-hop or contradiction-sensitive public decisions.
        if claim_scope != "single_document":
            return {
                "candidate": "MAYBE",
                "reason": "UNQUALIFIED_CONTEXT_SCOPE",
                "verdict_qualified": False,
                "raw_score": None,
            }
        hits = self.store.search(question, 10)
        selected = [{"page": h["data"]["url"], "text": h["data"]["text"]} for h in hits[:1]]
        result = self.model.predict(question, selected)
        return result | {
            "verdict_qualified": False,
            "calibrated_correctness": None,
            "context_document_ids": [h["id"] for h in hits[:1]],
            "remaining_evidence_document_ids": [h["id"] for h in hits[1:]],
            "evidence_policy": self.config["retrieval_policy"],
            "scope": "Historical single-document research candidate only; all other documents remain in the store for further checks.",
        }
