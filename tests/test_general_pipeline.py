import hashlib
import time

import pytest

from houndoracle import Claim, DocumentInput, Oracle
from houndoracle.calibration import Calibration, CalibrationBin
from houndoracle.compiler import compile_general_question, equivalent
from houndoracle.general import GeneralVerifier
from houndoracle.models import VerifyRequest


class Pairs:
    version = "fixture-not-real-neural-model"

    def __init__(self):
        self.calls = []

    def predict(self, question, passage):
        self.calls.append((question, passage))
        return {
            "candidate": "NO" if "refutation" in passage else "YES",
            "raw_score": 0.95,
            "truncated": False,
        }


def add(oracle, text="support evidence", **kw):
    now = time.time()
    return oracle.store.ingest(
        DocumentInput(
            url=kw.pop("url", "https://primary.example.org/a"),
            text=text,
            origin="primary.example.org",
            observed_at=now,
            expires_at=now + 100,
            **kw,
        )
    )


def test_general_representation_preserves_scope_and_exact_identity():
    q = "Is the reactor active in 2024?"
    assert compile_general_question(q).proposition == q
    assert not equivalent(
        compile_general_question(q), compile_general_question("Is the reactor active in 2025?")
    )
    assert not equivalent(
        compile_general_question(q), compile_general_question("Is the reactor inactive in 2024?")
    )
    assert equivalent(
        compile_general_question("Is it false that Is it false that " + q),
        compile_general_question(q),
    )
    assert compile_general_question("write a poem") is None
    with pytest.raises(ValueError):
        Claim(kind="prose", proposition=q, value="1")


def test_public_general_candidate_is_never_accepted_without_calibration():
    pairs = Pairs()
    oracle = Oracle(general_verifier=GeneralVerifier(pairs))
    try:
        add(oracle, reviewed=True)
        result = oracle.verify("Is the reactor active?")
        assert result.decision == "MAYBE" and result.reason == "UNCALIBRATED"
        assert (
            result.diagnostics.candidate_decision == "YES"
            and result.diagnostics.evidence_sufficient
        )
        assert result.confidence is None and result.citations
        assert (
            oracle.verify("Is it false that Is the reactor active?").diagnostics.candidate_decision
            == "NO"
        )
        assert (
            oracle.verify(
                "Is it false that Is it false that Is the reactor active?"
            ).diagnostics.candidate_decision
            == "YES"
        )
    finally:
        oracle.store.close()


def test_acquisition_is_not_human_review_and_snippets_cannot_qualify():
    oracle = Oracle(general_verifier=GeneralVerifier(Pairs()))
    try:
        add(oracle, retrieval_kind="search_snippet")
        assert not oracle.verify("Is the reactor active?").diagnostics.evidence_sufficient
        text = "support evidence"
        add(
            oracle,
            text,
            retrieval_kind="source_document",
            validation_status="acquired",
            validation_method="bounded-source-acquisition-v1",
            validation_content_hash=hashlib.sha256(text.encode()).hexdigest(),
        )
        result = oracle.verify("Is the reactor active?")
        assert result.diagnostics.candidate_decision == "YES" and result.decision == "MAYBE"
        assert all(
            not __import__("json").loads(r["data"])["reviewed"] for r in oracle.store.documents()
        )
        with pytest.raises(ValueError):
            add(
                oracle,
                "changed",
                retrieval_kind="source_document",
                validation_status="acquired",
                validation_method="bounded-source-acquisition-v1",
                validation_content_hash="0" * 64,
            )
    finally:
        oracle.store.close()


def test_conflicts_and_duplicates_are_not_votes_for_truth():
    oracle = Oracle(general_verifier=GeneralVerifier(Pairs()))
    try:
        add(oracle, reviewed=True)
        add(oracle, url="https://copy.example.org/b", reviewed=True)
        result = oracle.verify("Is the reactor active?")
        assert result.diagnostics.evidence_groups == 1
        add(oracle, "refutation evidence", url="https://other.test/a", reviewed=True)
        result = oracle.verify("Is the reactor active?")
        assert result.decision == "MAYBE" and result.reason == "CONTRADICTORY_EVIDENCE"
        assert result.diagnostics.contradiction_detected
    finally:
        oracle.store.close()


def test_online_search_full_source_budget_and_memoization():
    class Search:
        def __init__(self):
            self.queries = []

        def retrieve(self, q, budget):
            budget.reserve(0)
            self.queries.append(q)
            return [{"url": "https://example.org/a", "text": "untrusted snippet"}]

    def source(url, budget):
        budget.reserve(0)
        return {"url": url, "text": "support evidence"}

    pairs = Pairs()
    search = Search()
    oracle = Oracle(
        local_only=False,
        provider=search,
        general_verifier=GeneralVerifier(
            pairs, source_acquirer=source, acquirer_version="fixture-v1"
        ),
    )
    try:
        result = oracle.verify("Is the reactor active?", allow_network=True, max_requests=16)
        assert result.decision == "MAYBE" and result.reason == "UNCALIBRATED"
        assert result.diagnostics.candidate_decision == "YES"
        assert (
            result.diagnostics.search_actions == 4 and result.diagnostics.retrieved_documents == 1
        )
        assert result.usage["requests"] == 5 and len(pairs.calls) == 1
        assert len(set(search.queries)) == 4
    finally:
        oracle.store.close()


def test_network_failure_is_not_refutation_or_evidence():
    class Search:
        def retrieve(self, q, budget):
            budget.reserve(0)
            return [{"url": "https://example.org/a"}]

    def fail(url, budget):
        budget.reserve(0)
        raise OSError("private provider error")

    oracle = Oracle(
        local_only=False,
        provider=Search(),
        general_verifier=GeneralVerifier(Pairs(), source_acquirer=fail),
    )
    try:
        result = oracle.verify("Is the reactor active?", allow_network=True, max_requests=1)
        assert result.decision == "MAYBE" and result.reason == "BUDGET_EXHAUSTED"
        assert "private provider error" not in result.model_dump_json()
    finally:
        oracle.store.close()


def test_qualified_fixture_cache_invalidation_and_policy_binding():
    oracle = Oracle(general_verifier=GeneralVerifier(Pairs()))
    try:
        # Synthetic control-flow certificate only, never a benchmark claim.
        oracle.calibration = Calibration(
            policy_id=oracle.general_verifier.request_policy_id(
                oracle,
                VerifyRequest(question="Is the reactor active?"),
            ),
            population="synthetic-unit-test",
            supported_kinds=["prose", "not"],
            expires_at=time.time() + 100,
            dataset_digest="0" * 64,
            bins=[CalibrationBin(lower=0, upper=1, correct=100, count=100)],
        )
        doc = add(oracle, reviewed=True)
        result = oracle.verify("Is the reactor active?")
        assert result.decision == "YES" and result.confidence >= 0.8
        assert oracle.verify("Is the reactor active?").cache_hit
        changed = oracle.verify("Is the reactor active?", max_seconds=20)
        assert changed.decision == "MAYBE" and not changed.cache_hit
        oracle.store.revoke(doc)
        assert oracle.verify("Is the reactor active?").decision == "MAYBE"
        oracle.calibration.policy_id = "different-policy"
        add(oracle, reviewed=True)
        assert oracle.verify("Is the reactor active?").decision == "MAYBE"
    finally:
        oracle.store.close()


def test_stale_or_truncated_data_never_qualifies():
    pairs = Pairs()
    original = pairs.predict
    pairs.predict = lambda *a: original(*a) | {"truncated": True}
    oracle = Oracle(general_verifier=GeneralVerifier(pairs))
    try:
        add(oracle, reviewed=True)
        result = oracle.verify("Is the reactor active?")
        assert result.diagnostics.candidate_decision == "MAYBE"
        assert not result.diagnostics.evidence_sufficient
    finally:
        oracle.store.close()


def test_aggregator_preserves_uncertainty_without_duplicate_inflation():
    from houndoracle.general import GroupedAggregator

    head = GroupedAggregator()
    weak = {"candidate": "YES", "truncated": False, "raw_score": 0.6, "group": "g"}
    strong = weak | {"raw_score": 0.95}
    a = head.combine([weak])
    b = head.combine([strong])
    duplicate = head.combine([strong, strong | {"group": "copied-other-host"}])
    assert a["candidate"] == b["candidate"] == "YES"
    assert a["raw_score"] < b["raw_score"]
    assert duplicate["raw_score"] == b["raw_score"]
    assert head.combine([strong, strong | {"candidate": "NO"}])["candidate"] == "MAYBE"


def test_online_acceptance_waits_for_counter_evidence_acquisition():
    class Search:
        def retrieve(self, q, budget):
            budget.reserve(0)
            return [
                {
                    "url": "https://counter.test/no"
                    if "contradictory" in q
                    else "https://primary.test/yes"
                }
            ]

    acquired = []

    def source(url, budget):
        budget.reserve(0)
        acquired.append(url)
        return {
            "url": url,
            "text": "refutation evidence" if "counter" in url else "support evidence",
        }

    oracle = Oracle(
        local_only=False,
        provider=Search(),
        general_verifier=GeneralVerifier(
            Pairs(), source_acquirer=source, acquirer_version="counter-fixture"
        ),
    )
    try:
        request = VerifyRequest(
            question="Is the reactor active?", allow_network=True, max_requests=16
        )
        oracle.calibration = Calibration(
            policy_id=oracle.general_verifier.request_policy_id(oracle, request),
            population="synthetic-unit-test-only",
            supported_kinds=["prose"],
            expires_at=time.time() + 100,
            dataset_digest="0" * 64,
            bins=[CalibrationBin(lower=0, upper=1, correct=100, count=100)],
        )
        result = oracle.verify(request.question, allow_network=True, max_requests=16)
        assert "https://counter.test/no" in acquired
        assert result.decision == "MAYBE" and result.diagnostics.contradiction_detected
    finally:
        oracle.store.close()


def test_aggregator_retraining_changes_pipeline_identity():
    pipeline = GeneralVerifier(Pairs())
    before = pipeline.version
    pipeline.aggregator.model.fit([([1.0, 0.0, 0.0], 1), ([0.0, 1.0, 0.0], 0)], steps=1)
    assert pipeline.version != before


def test_source_revalidation_revokes_machine_qualification():
    from houndoracle.budget import Budget
    from houndoracle.freshness import refresh_due

    oracle = Oracle(general_verifier=GeneralVerifier(Pairs()))
    try:
        now = time.time()
        text = "support evidence"
        oracle.store.ingest(
            DocumentInput(
                url="https://example.org/a",
                text=text,
                origin="example.org",
                observed_at=now - 100,
                expires_at=now - 1,
                retrieval_kind="source_document",
                validation_status="acquired",
                validation_method="bounded-source-acquisition-v1",
                validation_content_hash=hashlib.sha256(text.encode()).hexdigest(),
            )
        )
        refresh_due(
            oracle.store,
            Budget(1, 10, 0),
            transport=lambda *a, **kw: {"status": 200, "body": b"changed evidence", "headers": {}},
        )
        import json

        doc = json.loads(oracle.store.documents()[0]["data"])
        assert (
            doc.get("validation_status", "unvalidated") == "unvalidated"
            and doc.get("validation_content_hash") is None
        )
        assert oracle.verify("Is the reactor active?").diagnostics.qualified_documents == 0
    finally:
        oracle.store.close()


def test_long_support_document_cannot_starve_counter_evidence():
    oracle = Oracle(general_verifier=GeneralVerifier(Pairs()))
    try:
        refute = add(oracle, "refutation evidence", url="https://counter.test/no", reviewed=True)
        long_text = "support evidence reactor active. " * 700
        # Force the long document first in the store's deterministic order.
        for i in range(100):
            url = f"https://primary.test/{i}"
            key = hashlib.sha256(
                (url + "\0" + hashlib.sha256(long_text.encode()).hexdigest()).encode()
            ).hexdigest()
            if key < refute:
                break
        add(oracle, long_text, url=url, reviewed=True)
        result = oracle.verify("Is the reactor active?")
        assert result.diagnostics.contradiction_detected
        assert result.decision == "MAYBE"
        assert any(c.document_id == refute for c in result.citations)
    finally:
        oracle.store.close()


def test_evidence_revoked_during_inference_cannot_be_accepted():
    pairs = Pairs()
    oracle = Oracle(general_verifier=GeneralVerifier(pairs))
    try:
        doc = add(oracle, reviewed=True)
        original = pairs.predict

        def revoke(*args):
            result = original(*args)
            oracle.store.revoke(doc)
            return result

        pairs.predict = revoke
        request = VerifyRequest(question="Is the reactor active?")
        oracle.calibration = Calibration(
            policy_id=oracle.general_verifier.request_policy_id(oracle, request),
            population="synthetic-unit-tests-only",
            supported_kinds=["prose"],
            expires_at=time.time() + 100,
            dataset_digest="0" * 64,
            bins=[CalibrationBin(lower=0, upper=1, correct=100, count=100)],
        )
        result = oracle.verify(request.question)
        assert result.decision == "MAYBE" and result.reason == "EVIDENCE_CHANGED_DURING_EVALUATION"
    finally:
        oracle.store.close()


def test_legacy_structured_and_default_document_wire_shapes_are_preserved(oracle, ingest):
    import json

    from houndoracle.models import EvaluationTrace

    claim = Claim(kind="release_year", entity="Widget", value="2024")
    assert "proposition" not in claim.model_dump()
    legacy = claim.model_dump() | {"proposition": None}
    ingest()
    result = oracle.verify("Verify structured proposition: " + json.dumps(legacy), claim=claim)
    assert result.decision == "YES"
    assert "evaluated_passages" not in EvaluationTrace().model_dump()
    document = json.loads(oracle.store.documents()[0]["data"])
    assert "validation_status" not in document


def test_cancellation_precedes_cached_general_answer_and_scope_is_refreshed():
    oracle = Oracle(general_verifier=GeneralVerifier(Pairs()))
    try:
        request = VerifyRequest(question="Is the reactor active?")
        oracle.calibration = Calibration(
            policy_id=oracle.general_verifier.request_policy_id(oracle, request),
            population="synthetic-unit-tests-only",
            supported_kinds=["prose", "not"],
            expires_at=time.time() + 100,
            dataset_digest="0" * 64,
            bins=[CalibrationBin(lower=0, upper=1, correct=100, count=100)],
        )
        add(oracle, reviewed=True)
        assert oracle.verify(request.question).decision == "YES"
        result = oracle.verify(request.question, _control=lambda: "JOB_CANCELLED")
        assert result.decision == "MAYBE" and result.reason == "JOB_CANCELLED"
        cached = oracle.verify("Is it false that Is it false that " + request.question)
        assert (
            cached.cache_hit
            and cached.claim.kind == "not"
            and cached.claim.children[0].kind == "not"
        )
        assert cached.diagnostics.evaluated_passages == 0
        assert set(cached.diagnostics.stage_seconds) == {"cache_lookup"}
    finally:
        oracle.store.close()
