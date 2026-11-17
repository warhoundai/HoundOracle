"""Experimental local text-CAPTCHA recognition; never a claim of server acceptance."""

import base64
import json
import math
import os
import signal
import string
import subprocess
import sys
import time

ALPHABET = string.ascii_letters + string.digits


def solve_text_captcha(
    image,
    *,
    backend="tesseract",
    expected_length=None,
    alphabet=ALPHABET,
    max_seconds=20,
    challenge_type="text",
):
    start = time.monotonic()
    result = {
        "status": "UNSUPPORTED_CHALLENGE",
        "candidate": None,
        "server_accepted": False,
        "backend": backend,
        "local_only": True,
        "network_requests": 0,
    }
    if challenge_type != "text":
        return result
    if backend not in {"ddddocr", "tesseract"}:
        raise ValueError("Choose ddddocr or tesseract")
    if not isinstance(image, bytes) or not 1 <= len(image) <= 1_000_000:
        raise ValueError("Image must contain 1..1000000 bytes")
    if expected_length is not None and (
        type(expected_length) is not int or not 1 <= expected_length <= 32
    ):
        raise ValueError("Expected text length must be 1..32")
    if (
        not isinstance(alphabet, str)
        or not 1 <= len(alphabet) <= 62
        or any(c not in ALPHABET for c in alphabet)
        or len(set(alphabet)) != len(alphabet)
    ):
        raise ValueError("Use a unique ASCII letter/digit alphabet")
    if (
        not isinstance(max_seconds, (int, float))
        or not math.isfinite(max_seconds)
        or not 0 < max_seconds <= 30
    ):
        raise ValueError("Time limit must be >0 and <=30 seconds")
    env = {
        k: os.environ[k]
        for k in ("PATH", "PYTHONPATH", "LANG", "LC_ALL", "TESSDATA_PREFIX")
        if k in os.environ
    }
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    payload = {
        "image": base64.b64encode(image).decode(),
        "backend": backend,
        "alphabet": alphabet,
        "seconds": max_seconds,
    }
    try:
        with subprocess.Popen(
            [sys.executable, "-m", "houndoracle.captcha_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=(os.name == "posix"),
        ) as proc:
            try:
                output, _ = proc.communicate(json.dumps(payload).encode(), timeout=max_seconds)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
                proc.communicate()
                raise
        if proc.returncode != 0 or len(output) > 65536:
            result["status"] = "BACKEND_FAILED"
        else:
            worker = json.loads(output)
            result["status"] = worker["status"]
            text = worker.get("candidate")
            if result["status"] == "CANDIDATE":
                result["status"] = "INVALID_CANDIDATE"
            if isinstance(text, str) and 1 <= len(text) <= 32 and all(c in alphabet for c in text):
                if expected_length is None or len(text) == expected_length:
                    result.update(status="CANDIDATE", candidate=text)
                else:
                    result["status"] = "LENGTH_MISMATCH"
            result["backend_version"] = worker.get("backend_version")
    except subprocess.TimeoutExpired:
        result["status"] = "TIME_BUDGET_EXCEEDED"
    except (ValueError, KeyError, OSError):
        result["status"] = "BACKEND_FAILED"
    result["latency_seconds"] = time.monotonic() - start
    return result


def solve_encoded_captcha(image_base64, **options):
    if not isinstance(image_base64, str) or not 1 <= len(image_base64) <= 1_333_336:
        raise ValueError("Bounded base64 image required")
    try:
        image = base64.b64decode(image_base64, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid image base64") from exc
    return solve_text_captcha(image, **options)
