"""Opt-in process boundary to a separately AGPL-licensed local Google v2 solver."""

import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit


def solve_google_recaptcha(
    target_url, *, helper_directory, model_directory, port=9222, max_seconds=90, allow_network=False
):
    if not allow_network:
        raise PermissionError("Google widget interaction requires explicit network permission")
    p = urlsplit(target_url)
    if (
        p.scheme != "https"
        or p.netloc != "www.google.com"
        or p.username
        or p.password
        or p.fragment
        or len(target_url) > 4096
        or any(c in target_url for c in ("\r", "\n", "\x00"))
    ):
        raise ValueError("An exact existing https://www.google.com tab URL is required")
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("A loopback CDP port in1024..65535 is required")
    if (
        not isinstance(max_seconds, (float, int))
        or not math.isfinite(max_seconds)
        or not 0 < max_seconds <= 120
    ):
        raise ValueError("Solver wall-clock limit must be >0 and <=120seconds")
    helper = Path(helper_directory).resolve()
    if not (helper / "bridge.mjs").is_file():
        raise ValueError("Install the separate recaptcha-local helper first")
    models = Path(model_directory).resolve()
    if not models.is_dir():
        return {"status": "MODELS_UNAVAILABLE", "widget_solved": False, "server_accepted": None}
    payload = {"target_url": target_url, "model_dir": str(models), "port": port}
    env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL") if k in os.environ}
    env.update({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    start = time.monotonic()
    try:
        with subprocess.Popen(
            ["node", str(helper / "bridge.mjs")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=os.name == "posix",
            cwd=helper,
        ) as proc:
            try:
                output, _ = proc.communicate(json.dumps(payload).encode(), timeout=max_seconds)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
                proc.communicate()
                return {
                    "status": "TIMEOUT_OUTCOME_UNKNOWN",
                    "widget_solved": False,
                    "server_accepted": None,
                    "retry_automatically": False,
                }
        if proc.returncode != 0 or len(output) > 65536:
            raise ValueError("Helper failed")
        result = json.loads(output)
        if not isinstance(result, dict) or result.get("status") not in {
            "WIDGET_SOLVED",
            "UNVERIFIED",
            "SOLVER_FAILED",
        }:
            raise ValueError("Invalid helper response")
        # Never forward response tokens, cookies, raw errors or page content.
        clean = {
            k: v
            for k, v in result.items()
            if k
            in {
                "status",
                "widget_solved",
                "verification",
                "captcha_type",
                "image_challenge_solved",
                "attempts",
                "local_inference",
                "hosted_solver_calls",
                "challenge_image_requests",
                "challenge_image_bytes",
                "failure_stage",
                "failure_reason",
                "failure_type",
            }
        }
        clean["server_accepted"] = None
        clean["latency_seconds"] = time.monotonic() - start
        return clean
    except (OSError, ValueError, KeyError):
        return {
            "status": "SOLVER_FAILED",
            "widget_solved": False,
            "server_accepted": None,
            "latency_seconds": time.monotonic() - start,
        }
