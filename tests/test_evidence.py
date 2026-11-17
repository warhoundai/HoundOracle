import json
import time

from houndoracle import DocumentInput, Oracle
from houndoracle.compiler import compile_question
from houndoracle.evaluators import features


def test_uncalibrated_abstains():
    o = Oracle()
    now = time.time()
    o.store.ingest(
        DocumentInput(
            url="file:official",
            text="Widget was released in 2024.",
            origin="official",
            observed_at=now,
            expires_at=now + 100,
            reviewed=True,
            primary=True,
        )
    )
    assert o.verify("Was Widget released in 2024?").reason == "UNCALIBRATED"


def test_cache_exact_and_dependency_invalidation(oracle, ingest):
    original = ingest()
    first = oracle.verify("Was Widget released in 2024?")
    assert first.decision == "YES" and not first.cache_hit
    second = oracle.verify("Was Widget released in 2024?")
    assert second.cache_hit
    assert not oracle.verify("Was Widget released in 2023?").cache_hit
    assert oracle.explain(first.result_id)["valid"]
    ingest("Widget was released in 2025.")
    assert not oracle.explain(first.result_id)["valid"]
    assert oracle.verify("Was Widget released in 2024?").decision == "NO"
    assert not oracle.store.document(original)["active"]


def test_contradictory_new_source(oracle, ingest):
    ingest()
    oracle.verify("Was Widget released in 2024?")
    ingest("Widget was released in 2025.", url="https://other.example/record", origin="other")
    result = oracle.verify("Was Widget released in 2024?")
    assert result.decision == "MAYBE" and not result.cache_hit


def test_duplicates_do_not_increase_independence(oracle, ingest):
    ingest()
    ingest(url="https://copy.example/article", origin="copy")
    rels = oracle.compiler.evaluate(
        compile_question("Was Widget released in 2024?"), oracle.store.documents()
    )
    assert len(rels) == 2
    assert len({r.citation.group for r in rels}) == 1
    assert features(rels)[0] < 1


def test_origin_and_upstream_transitive_grouping(oracle, ingest):
    a = ingest()
    ingest(
        "Widget was released in 2024.\n",
        url="https://syndicated.example",
        origin="syndicated",
        upstream_ids=[a],
    )
    rels = oracle.compiler.evaluate(
        compile_question("Was Widget released in 2024?"), oracle.store.documents()
    )
    assert len({r.citation.group for r in rels}) == 1
    oracle.store.revoke(a)
    assert not oracle.store.documents()


def test_expired_parent_invalidates_child(oracle, ingest):
    a = ingest(expires_at=time.time() + 0.1)
    b = ingest("Widget was released in 2024.\n", url="file:derived", upstream_ids=[a])
    result = oracle.verify("Was Widget released in 2024?")
    assert result.decision == "YES"
    assert not oracle.store._fresh(b, time.time() + 1)
    assert oracle.store.cached("unused") is None


def test_snapshot_drops_trust(oracle, ingest):
    ingest(public=True)
    snapshot = oracle.store.export()
    other = Oracle()
    other.store.import_snapshot(snapshot)
    row = other.store.documents()[0]
    data = json.loads(row["data"])
    assert not data["reviewed"] and not data["primary"] and not data["public"]
    assert other.verify("Was Widget released in 2024?").decision == "MAYBE"


def test_expired_calibration_cannot_reuse_cache(oracle, ingest):
    ingest()
    oracle.verify("Was Widget released in 2024?")
    oracle.calibration.expires_at = time.time() - 1
    result = oracle.verify("Was Widget released in 2024?")
    assert result.decision == "MAYBE" and not result.cache_hit


def test_unreviewed_source_is_not_authority(oracle, ingest):
    ingest(reviewed=False)
    assert oracle.verify("Was Widget released in 2024?").decision == "MAYBE"


def test_model_change_invalidates_calibration(oracle, ingest):
    ingest()
    oracle.aggregator.bias = 1
    assert oracle.verify("Was Widget released in 2024?").decision == "MAYBE"


def test_cycle_rejected_transactionally(oracle, ingest):
    a = ingest()
    b = ingest("Widget was released in 2024.\n", url="file:b", upstream_ids=[a])
    import pytest

    with pytest.raises(ValueError, match="Cyclic"):
        ingest(upstream_ids=[b])
    assert oracle.store.document(a)["active"] and oracle.store.document(b)["active"]


def test_cached_answer_obeys_deadline(oracle, ingest):
    ingest()
    oracle.verify("Was Widget released in 2024?")
    result = oracle.verify("Was Widget released in 2024?", max_seconds=1e-12)
    assert result.decision == "MAYBE" and result.reason == "TIME_EXHAUSTED"


def test_local_snapshot_preserves_provenance(oracle, ingest):
    parent = ingest()
    ingest("Widget was released in 2024.\n", url="file:derived", upstream_ids=[parent])
    other = Oracle()
    other.store.import_snapshot(oracle.store.export())
    assert len(other.store.documents()) == 2
    other.store.revoke(parent)
    assert not other.store.documents()
