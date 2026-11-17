"""Private checkpoint integrity, object scope and conditional writes."""

import hashlib
import io
import json
import urllib.error
import zlib

import pytest

from houndoracle.experiments.checkpoint import CheckpointError, SignedS3Checkpoint

URL = "https://private.s3.us-east-1.amazonaws.com/run/journal?X-Amz-Date=20261116T060000Z&X-Amz-Expires=900&X-Amz-Signature=test&X-Amz-Credential=test"


class Response(io.BytesIO):
    def __init__(self, data=b""):
        super().__init__(data)
        self.headers = {"ETag": '"version1"'}


def backend(opener):
    return SignedS3Checkpoint(
        bucket="private", key="run/journal", get_url=URL, put_url=URL, opener=opener
    )


def test_restore_digest_and_private_permissions(tmp_path):
    raw = b"SQLite format 3\x00" + b"data"
    payload = (
        json.dumps({"schema": 1, "sha256": hashlib.sha256(raw).hexdigest()}).encode()
        + b"\n"
        + zlib.compress(raw)
    )
    checkpoint = backend(lambda *a, **k: Response(payload))
    target = tmp_path / "journal"
    assert checkpoint.restore(target) and target.read_bytes() == raw
    assert target.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        checkpoint.restore(target)
    with pytest.raises(CheckpointError):
        backend(lambda *a, **k: Response(payload[:-2])).restore(tmp_path / "corrupt")


def test_conditional_commit_and_no_conflict_retry(tmp_path):
    requests = []

    def open_request(request, **kwargs):
        requests.append(request)
        if len(requests) == 3:
            raise urllib.error.HTTPError(URL, 412, "conflict", {}, None)
        return Response()

    checkpoint = backend(open_request)

    class Journal:
        def snapshot(self, path):
            path.write_bytes(b"SQLite format 3\x00data")

    checkpoint.commit(Journal())
    checkpoint.commit(Journal())
    assert requests[0].get_header("If-none-match") == "*"
    assert requests[1].get_header("If-match") == '"version1"'
    with pytest.raises(CheckpointError, match="412"):
        checkpoint.commit(Journal())
    assert len(requests) == 3
    with pytest.raises(ValueError):
        SignedS3Checkpoint(bucket="other", key="run/journal", get_url=URL, put_url=URL)
