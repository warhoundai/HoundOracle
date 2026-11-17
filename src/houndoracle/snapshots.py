"""Opt-in signed S3 snapshots. No account, bucket or billable service is created.

The operator supplies an existing private, versioned bucket and rights-approved
public evidence. Private text, questions and answers are never part of a snapshot.
"""

import base64
import hashlib
import json
import time

from .models import DocumentInput


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def prepare(store, *, allowed_licenses, lifetime_seconds=86400):
    if not 1 <= lifetime_seconds <= 604800:
        raise ValueError("Snapshot lifetime must be at most seven days")
    documents, blobs = [], {}
    candidates = {}
    for row in store.documents():
        doc = json.loads(row["data"])
        if doc["public"] and doc["reviewed"] and doc["license"] in allowed_licenses:
            candidates[row["id"]] = doc
    # Never leak private dependency identifiers or detach dependent evidence.
    while True:
        removed = [
            key
            for key, doc in candidates.items()
            if not set(doc["upstream_ids"]).issubset(candidates)
        ]
        if not removed:
            break
        for key in removed:
            del candidates[key]
    completed = set()
    while candidates:
        ready = sorted(
            key for key, doc in candidates.items() if set(doc["upstream_ids"]).issubset(completed)
        )
        if not ready:
            raise ValueError("Cyclic evidence cannot be shared")
        for key in ready:
            doc = candidates.pop(key)
            blob = doc.pop("text").encode()
            digest = hashlib.sha256(blob).hexdigest()
            blobs[digest] = blob
            if len(documents) >= 1000 or sum(len(value) for value in blobs.values()) > 20_000_000:
                raise ValueError("Snapshot exceeds resource bounds")
            documents.append({"id": key, "metadata": doc, "sha256": digest})
            completed.add(key)
    now = time.time()
    manifest = {
        "schema_version": 1,
        "issued_at": now,
        "expires_at": now + lifetime_seconds,
        "documents": documents,
    }
    return manifest, blobs


def signed_manifest(manifest, private_key):
    data = canonical_json(manifest)
    return canonical_json(
        {"manifest": manifest, "signature": base64.b64encode(private_key.sign(data)).decode()}
    )


def verify_manifest(data, public_key):
    if len(data) > 2_000_000:
        raise ValueError("Manifest too large")
    envelope = json.loads(data)
    if set(envelope) != {"manifest", "signature"}:
        raise ValueError("Invalid envelope")
    manifest = envelope["manifest"]
    public_key.verify(
        base64.b64decode(envelope["signature"], validate=True), canonical_json(manifest)
    )
    now = time.time()
    if (
        manifest["schema_version"] != 1
        or not manifest["issued_at"] <= now < manifest["expires_at"]
        or manifest["expires_at"] - manifest["issued_at"] > 604800
        or len(manifest["documents"]) > 1000
    ):
        raise ValueError("Unsupported, expired or oversized snapshot")
    completed = set()
    for item in manifest["documents"]:
        if len(item["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in item["sha256"]):
            raise ValueError("Invalid object digest")
        expected = hashlib.sha256(
            (item["metadata"]["url"] + "\0" + item["sha256"]).encode()
        ).hexdigest()
        if item["id"] != expected or item["id"] in completed:
            raise ValueError("Invalid document identity")
        if not set(item["metadata"]["upstream_ids"]).issubset(completed):
            raise ValueError("Missing or cyclic upstream dependency")
        completed.add(item["id"])
    return manifest


class S3Snapshots:
    def __init__(self, client, bucket, *, prefix="houndoracle"):
        if not prefix or ".." in prefix or prefix.startswith("/"):
            raise ValueError("Invalid object prefix")
        self.client, self.bucket, self.prefix = client, bucket, prefix.rstrip("/")

    def check_bucket(self):
        version = self.client.get_bucket_versioning(Bucket=self.bucket)
        block = self.client.get_public_access_block(Bucket=self.bucket)[
            "PublicAccessBlockConfiguration"
        ]
        required = {
            "BlockPublicAcls",
            "IgnorePublicAcls",
            "BlockPublicPolicy",
            "RestrictPublicBuckets",
        }
        if version.get("Status") != "Enabled" or not all(block.get(k) for k in required):
            raise ValueError("Snapshots require a private versioned bucket")

    def publish(
        self, manifest, blobs, private_key, *, authorized=False, allowed_licenses=frozenset()
    ):
        if not authorized:
            raise PermissionError("Central sharing requires explicit authorization")
        data = signed_manifest(manifest, private_key)
        verify_manifest(data, private_key.public_key())
        expected_blobs = {item["sha256"] for item in manifest["documents"]}
        if expected_blobs != set(blobs):
            raise ValueError("Snapshot blob set mismatch")
        for item in manifest["documents"]:
            metadata = item["metadata"]
            if not (
                metadata["public"]
                and metadata["reviewed"]
                and metadata["license"] in allowed_licenses
            ):
                raise PermissionError("Unapproved evidence or license")
            DocumentInput.model_validate({**metadata, "text": blobs[item["sha256"]].decode()})
        self.check_bucket()
        for digest, blob in blobs.items():
            if hashlib.sha256(blob).hexdigest() != digest:
                raise ValueError("Blob digest mismatch")
            # Conditional put deduplicates immutable content across snapshots.
            try:
                self.client.put_object(
                    Bucket=self.bucket,
                    Key=f"{self.prefix}/blobs/{digest}",
                    Body=blob,
                    ServerSideEncryption="AES256",
                    IfNoneMatch="*",
                )
            except Exception as exc:
                if (
                    getattr(exc, "response", {}).get("ResponseMetadata", {}).get("HTTPStatusCode")
                    != 412
                ):
                    raise
        data = signed_manifest(manifest, private_key)
        digest = hashlib.sha256(data).hexdigest()
        response = self.client.put_object(
            Bucket=self.bucket,
            Key=f"{self.prefix}/snapshots/{digest}.json",
            Body=data,
            ServerSideEncryption="AES256",
            ContentType="application/json",
            IfNoneMatch="*",
        )
        return {"sha256": digest, "version_id": response.get("VersionId")}

    def download(self, digest, public_key, store, *, authorized=False, trust_remote_review=False):
        if not authorized:
            raise PermissionError("Central access requires explicit authorization")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid manifest digest")
        response = self.client.get_object(
            Bucket=self.bucket, Key=f"{self.prefix}/snapshots/{digest}.json"
        )
        data = response["Body"].read(2_000_001)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Manifest digest mismatch")
        manifest = verify_manifest(data, public_key)
        total, imports = 0, []
        # Validate every blob before mutating the local store.
        for item in manifest["documents"]:
            response = self.client.get_object(
                Bucket=self.bucket, Key=f"{self.prefix}/blobs/{item['sha256']}"
            )
            blob = response["Body"].read(1_000_001)
            total += len(blob)
            if (
                len(blob) > 1_000_000
                or total > 20_000_000
                or hashlib.sha256(blob).hexdigest() != item["sha256"]
            ):
                raise ValueError("Invalid or oversized evidence blob")
            doc = DocumentInput.model_validate(
                {
                    **item["metadata"],
                    "text": blob.decode(),
                    "reviewed": bool(trust_remote_review),
                    "public": False,
                }
            )
            if not item["metadata"].get("public") or not item["metadata"].get("reviewed"):
                raise ValueError("Snapshot contains unapproved material")
            imports.append(doc)
        with store.lock, store.db:
            return [store.ingest(doc) for doc in imports]
