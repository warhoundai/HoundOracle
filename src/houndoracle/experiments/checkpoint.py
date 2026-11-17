"""Opt-in private S3 journal snapshots using short-lived, object-scoped URLs; no credential export."""

import hashlib
import json
import os
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import zlib
from datetime import UTC, datetime
from pathlib import Path


class CheckpointError(RuntimeError):
    pass


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def signed_object(url, bucket, key):
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.port not in [None, 443]
    ):
        raise ValueError("Checkpoint requires an HTTPS S3 signed URL")
    if not host.endswith(".amazonaws.com") or not (
        ".s3." in host or ".s3-" in host or host.startswith("s3.")
    ):
        raise ValueError("Checkpoint host is not a permitted S3 endpoint")
    if ".s3" in host and not host.startswith("s3."):
        actual_bucket = host.split(".s3", 1)[0]
        actual_key = urllib.parse.unquote(parsed.path.lstrip("/"))
    else:
        pieces = urllib.parse.unquote(parsed.path.lstrip("/")).split("/", 1)
        actual_bucket = pieces[0]
        actual_key = pieces[1] if len(pieces) == 2 else ""
    if (actual_bucket, actual_key) != (bucket, key):
        raise ValueError("Signed URL does not match the explicit checkpoint object scope")
    query = urllib.parse.parse_qs(parsed.query)
    if not all(
        query.get(k) for k in ["X-Amz-Date", "X-Amz-Expires", "X-Amz-Signature", "X-Amz-Credential"]
    ):
        raise ValueError("Checkpoint must use an expiring signed request")
    duration = int(query["X-Amz-Expires"][0])
    if not 1 <= duration <= 900:
        raise ValueError("Checkpoint URL exceeds the broker time envelope")
    return (
        datetime.strptime(query["X-Amz-Date"][0], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).timestamp()
        + duration
    )


class SignedS3Checkpoint:
    def __init__(
        self,
        *,
        bucket,
        key,
        get_url,
        put_url,
        opener=None,
        max_compressed_bytes=4 * 1024 * 1024,
        max_raw_bytes=64 * 1024 * 1024,
    ):
        self.expires_at = min(
            signed_object(get_url, bucket, key), signed_object(put_url, bucket, key)
        )
        self.get_url, self.put_url = get_url, put_url
        self.etag = None
        self.lock = threading.RLock()
        self.open = opener or urllib.request.build_opener(NoRedirects()).open
        self.max_compressed, self.max_raw = max_compressed_bytes, max_raw_bytes

    def _request(self, request):
        try:
            return self.open(request, timeout=15)
        except urllib.error.HTTPError as exc:
            raise CheckpointError(
                f"Private checkpoint HTTP operation failed ({exc.code}); no automatic overwrite/retry"
            ) from None
        except urllib.error.URLError:
            raise CheckpointError(
                "Private checkpoint transport failed; signed URLs omitted"
            ) from None

    def restore(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(
                "Restore into a new private path, never overwrite an active journal"
            )
        with self.lock:
            try:
                response = self.open(urllib.request.Request(self.get_url, method="GET"), timeout=15)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    self.etag = None
                    return False
                raise CheckpointError(f"Private checkpoint read failed ({exc.code})") from None
            except urllib.error.URLError:
                raise CheckpointError("Private checkpoint read transport failed") from None
            with response:
                payload = response.read(self.max_compressed + 1)
                etag = response.headers.get("ETag")
            if len(payload) > self.max_compressed or not etag:
                raise CheckpointError("Checkpoint size/header validation failed")
            try:
                header, compressed = payload.split(b"\n", 1)
                metadata = json.loads(header)
                decoder = zlib.decompressobj()
                raw = decoder.decompress(compressed, self.max_raw + 1)
                if not decoder.eof or decoder.unused_data or len(raw) > self.max_raw:
                    raise ValueError("Invalid compressed envelope")
                if (
                    metadata.get("schema") != 1
                    or hashlib.sha256(raw).hexdigest() != metadata["sha256"]
                    or not raw.startswith(b"SQLite format 3\x00")
                ):
                    raise ValueError("Invalid snapshot digest/type")
            except (ValueError, KeyError, zlib.error):
                raise CheckpointError("Checkpoint content integrity validation failed") from None
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            self.etag = etag
            return True

    def commit(self, journal):
        with self.lock:
            fd, name = tempfile.mkstemp(suffix=".sqlite")
            os.close(fd)
            path = Path(name)
            try:
                journal.snapshot(path)
                if path.stat().st_size > self.max_raw:
                    raise CheckpointError("Journal exceeds raw snapshot limit")
                raw = path.read_bytes()
                header = json.dumps(
                    {"schema": 1, "sha256": hashlib.sha256(raw).hexdigest()}, sort_keys=True
                ).encode()
                payload = header + b"\n" + zlib.compress(raw, 3)
                if len(payload) > self.max_compressed:
                    raise CheckpointError("Journal exceeds compressed checkpoint limit")
                headers = {
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(len(payload)),
                }
                headers["If-Match" if self.etag else "If-None-Match"] = self.etag or "*"
                with self._request(
                    urllib.request.Request(
                        self.put_url, data=payload, headers=headers, method="PUT"
                    )
                ) as response:
                    etag = response.headers.get("ETag")
                if not etag:
                    raise CheckpointError(
                        "Checkpoint acknowledgement has no version; halt before provider work"
                    )
                self.etag = etag
            finally:
                path.unlink(missing_ok=True)
