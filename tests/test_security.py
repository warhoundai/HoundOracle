import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from houndoracle.budget import Budget, BudgetExceeded
from houndoracle.freshness import refresh_due
from houndoracle.network import HoundSearchAdapter, RetrievalError, addresses, fetch
from houndoracle.snapshots import S3Snapshots, prepare, signed_manifest, verify_manifest


@pytest.mark.parametrize(
    "ip",
    ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "fd00::1", "192.168.0.1", "100.64.0.1"],
)
def test_private_dns_blocked(monkeypatch, ip):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", (ip, 443))])
    with pytest.raises(RetrievalError, match="PRIVATE_NETWORK_BLOCKED"):
        addresses("attacker.example", 443)


def test_mixed_dns_blocked(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", (x, 443)) for x in ["8.8.8.8", "127.0.0.1"]],
    )
    with pytest.raises(RetrievalError):
        addresses("attacker.example", 443)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "https://user:password@example.org",
        "https://example.org:22/a",
        "http://127.0.0.1/a",
    ],
)
def test_unsafe_urls(url):
    with pytest.raises((RetrievalError, ValueError)):
        fetch(url, timeout=0.1)


def test_budget_atomic_reservations():
    budget = Budget(4, 10, 0.03)

    def reserve(_):
        try:
            budget.reserve(0.01)
            return True
        except BudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(reserve, range(20))) == 3
    assert budget.requests == 3
    assert budget.cost <= 0.03000001


def test_no_network_by_default(oracle):
    class Provider:
        def retrieve(self, *args):
            raise AssertionError("Network must not be used")

    oracle.local_only = False  # Explicit opt-in for legacy transport tests.
    oracle.provider = Provider()
    assert oracle.verify("Was Widget released in 2024?").decision == "MAYBE"


def test_provider_errors_and_exhaustion(oracle):
    class Provider:
        def retrieve(self, query, budget):
            budget.reserve(0.1)
            raise RuntimeError("secret-api-key")

    oracle.local_only = False  # Explicit opt-in for legacy transport tests.
    oracle.provider = Provider()
    result = oracle.verify("Was Widget released in 2024?", allow_network=True)
    assert result.reason == "BUDGET_EXHAUSTED" and result.usage["requests"] == 0
    result = oracle.verify("Was Widget released in 2024?", allow_network=True, max_cost_usd=0.1)
    assert result.reason == "RETRIEVAL_FAILED" and "secret" not in result.model_dump_json()
    assert result.decision == "MAYBE"


def test_live_evidence_is_unreviewed(oracle):
    class Provider:
        def retrieve(self, query, budget):
            budget.reserve(0)
            return [{"url": "https://example.org", "text": "Widget was released in 2024."}]

    oracle.local_only = False  # Explicit opt-in for legacy transport tests.
    oracle.provider = Provider()
    result = oracle.verify("Was Widget released in 2024?", allow_network=True)
    assert result.decision == "MAYBE"
    assert result.usage["requests"] == 1
    assert not json.loads(oracle.store.documents()[0]["data"])["reviewed"]


def test_remote_adapter_requires_tls():
    with pytest.raises(ValueError):
        HoundSearchAdapter("http://example.org", "secret", request_cost_upper_usd=0)


def test_refresh_unchanged_and_changed(oracle, ingest):
    doc_id = ingest(expires_at=time.time() + 0.01, etag="original")

    def unchanged(url, **kwargs):
        assert kwargs["headers"]["If-None-Match"] == "original"
        return {"status": 304, "headers": {}, "body": b""}

    time.sleep(0.02)
    result = refresh_due(oracle.store, Budget(1, 1, 0), transport=unchanged)
    assert result[0]["status"] == "unchanged"
    assert oracle.verify("Was Widget released in 2024?").decision == "YES"
    # Explicitly expire the same version again.
    data = json.loads(oracle.store.document(doc_id)["data"])
    data.update(observed_at=time.time() - 20, expires_at=time.time() - 10)
    from houndoracle import DocumentInput

    oracle.store.ingest(DocumentInput.model_validate(data))

    def changed(url, **kwargs):
        return {"status": 200, "headers": {}, "body": b"Widget was released in 2025."}

    result = refresh_due(oracle.store, Budget(1, 1, 0), transport=changed)
    assert result[0]["status"] == "review_required"
    assert oracle.verify("Was Widget released in 2025?").decision == "MAYBE"


def test_signed_public_snapshots(oracle, ingest):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    ingest(public=True, license="CC0-1.0")
    ingest("Private was released in 2024.", url="file:private", origin="private")
    manifest, blobs = prepare(oracle.store, allowed_licenses={"CC0-1.0"})
    assert len(manifest["documents"]) == 1 and len(blobs) == 1
    data = signed_manifest(manifest, key)
    assert verify_manifest(data, key.public_key()) == manifest
    damaged = json.loads(data)
    damaged["manifest"]["documents"][0]["metadata"]["origin"] = "attacker"
    from cryptography.exceptions import InvalidSignature

    with pytest.raises(InvalidSignature):
        verify_manifest(json.dumps(damaged).encode(), key.public_key())
    assert "Private" not in data.decode()


def test_snapshot_publish_requires_opt_in():
    store = S3Snapshots(object(), "bucket")
    with pytest.raises(PermissionError):
        store.publish({}, {}, object())
    with pytest.raises(PermissionError):
        store.download("0" * 64, object(), object())


def test_offline_verification_with_network_blocked(oracle, ingest, monkeypatch):
    ingest()

    def blocked(*args, **kwargs):
        raise AssertionError("Offline verification attempted network access")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    assert oracle.verify("Was Widget released in 2024?").decision == "YES"


def test_snapshot_excludes_private_dependencies(oracle, ingest):
    parent = ingest(public=False, license="CC0-1.0")
    ingest(
        "Widget was released in 2024.\n",
        url="file:derived",
        public=True,
        license="CC0-1.0",
        upstream_ids=[parent],
    )
    manifest, blobs = prepare(oracle.store, allowed_licenses={"CC0-1.0"})
    assert not manifest["documents"] and not blobs


def test_s3_roundtrip_preserves_dependencies(oracle, ingest):
    import io

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from houndoracle import Oracle

    class Client:
        def __init__(self):
            self.objects = {}

        def get_bucket_versioning(self, **kwargs):
            return {"Status": "Enabled"}

        def get_public_access_block(self, **kwargs):
            return {
                "PublicAccessBlockConfiguration": dict.fromkeys(
                    [
                        "BlockPublicAcls",
                        "IgnorePublicAcls",
                        "BlockPublicPolicy",
                        "RestrictPublicBuckets",
                    ],
                    True,
                )
            }

        def put_object(self, **kwargs):
            assert kwargs["ServerSideEncryption"] == "AES256"
            self.objects[kwargs["Key"]] = kwargs["Body"]
            return {"VersionId": "v1"}

        def get_object(self, **kwargs):
            return {"Body": io.BytesIO(self.objects[kwargs["Key"]])}

    parent = ingest(public=True, license="CC0-1.0")
    ingest(
        "Widget was released in 2024.\n",
        url="file:public-derived",
        origin="other",
        public=True,
        license="CC0-1.0",
        upstream_ids=[parent],
    )
    client = Client()
    snapshots = S3Snapshots(client, "test-bucket")
    key = Ed25519PrivateKey.generate()
    manifest, blobs = prepare(oracle.store, allowed_licenses={"CC0-1.0"})
    published = snapshots.publish(
        manifest, blobs, key, authorized=True, allowed_licenses={"CC0-1.0"}
    )
    local = Oracle()
    ids = snapshots.download(published["sha256"], key.public_key(), local.store, authorized=True)
    assert len(ids) == 2
    assert not json.loads(local.store.document(ids[0])["data"])["reviewed"]
    local.store.revoke(parent)
    assert not local.store.documents()


def test_houndsearch_contract_and_reservation(monkeypatch):
    from houndoracle import Oracle, network

    calls = []

    def transport(url, **kwargs):
        payload = json.loads(kwargs["body"])
        assert payload["budget"]["max_cost_usd"] == 0.01
        assert payload["budget"]["max_requests"] == 4
        assert payload["limit"] == 2
        calls.append(url)
        return {
            "body": json.dumps(
                {
                    "results": [
                        {
                            "url": "https://example.org/record",
                            "content": {"text": "Widget was released in 2024."},
                        }
                    ]
                }
            ).encode()
        }

    monkeypatch.setattr(network, "fetch", transport)
    provider = HoundSearchAdapter(
        "https://search.example.org", "secret", request_cost_upper_usd=0.01
    )
    oracle = Oracle(provider=provider, local_only=False)
    result = oracle.verify("Was Widget released in 2024?", allow_network=True, max_cost_usd=0.01)
    assert result.decision == "MAYBE" and len(calls) == 1
    assert oracle.store.documents()[0]["url"] == "https://example.org/record"
    assert result.usage["reserved_cost_usd"] == 0.01
    with pytest.raises(ValueError):
        HoundSearchAdapter("https://search.example.org", "secret", request_cost_upper_usd=0)
