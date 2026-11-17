import time

from houndoracle.models import DocumentInput
from houndoracle.store import Store


def document(store, name, text, *, expired=False, upstream=()):
    now = time.time()
    return store.ingest(
        DocumentInput(
            url=f"https://example.org/{name}",
            text=text,
            origin="example.org",
            observed_at=now - 100,
            expires_at=now - 1 if expired else now + 100,
            upstream_ids=list(upstream),
        )
    )


def test_expired_top_hits_do_not_hide_fresh_lower_ranked_evidence():
    store = Store()
    try:
        for i in range(15):
            document(store, f"old-{i}", "reactor active", expired=True)
        expected = document(store, "fresh", "reactor active " + "background " * 100)
        hits = store.search("reactor active", limit=1)
        assert [h["id"] for h in hits] == [expected]
    finally:
        store.close()


def test_search_rejects_expired_dependency_and_continues_ranking():
    store = Store()
    try:
        parent = document(store, "parent", "upstream record", expired=True)
        document(store, "derived", "reactor active", upstream=[parent])
        expected = document(store, "independent", "reactor active background context")
        assert [h["id"] for h in store.search("reactor active", 1)] == [expected]
        assert store.search("upstream", 1) == []
    finally:
        store.close()
