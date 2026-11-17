"""Bounded offline reasoning-model comparison with a verified temporary RAM checkpoint."""

import hashlib
import importlib.metadata
import inspect
import json
import os
import random
import resource
import shutil
import socket
import time
import urllib.request
from pathlib import Path

from houndoracle.experiments.completion_control import chat_with_stopping
from houndoracle.experiments.draft_verifier import predict

PRIVATE = Path("/workspace/houndoracle-data/campaign-40")
PUBLIC = Path("research/experiments/campaign-40")
MODEL = Path("/dev/shm/houndoracle-reasoning40/model.gguf")


def write_progress(value):
    (PRIVATE / "progress.json").write_text(json.dumps(value) + "\n")


def blocked_connect(*args, **kwargs):
    raise PermissionError("MODEL_INFERENCE_NETWORK_DISABLED")


def main():
    from llama_cpp import Llama, StoppingCriteriaList

    if "stopping_criteria" not in inspect.signature(Llama.create_completion).parameters:
        raise RuntimeError("Unsupported completion API; do not acquire a model")
    required = {"messages", "max_tokens", "temperature", "top_p", "seed", "response_format"}
    if not required.issubset(inspect.signature(Llama.create_chat_completion).parameters):
        raise RuntimeError("Unsupported chat API; do not acquire a model")
    protocol = json.loads((PUBLIC / "protocol.json").read_text())
    spec = protocol["model"]
    if (PRIVATE / "result.json").exists():
        raise FileExistsError("Preserve completed screen")
    if (
        hashlib.file_digest((PRIVATE / "inputs.json").open("rb"), "sha256").hexdigest()
        != protocol["inputs_sha256"]
    ):
        raise ValueError("Frozen cohort changed")
    inputs = json.loads((PRIVATE / "inputs.json").read_text())
    if len(inputs["questions"]) != 12 or len({r["group_id"] for r in inputs["questions"]}) != 12:
        raise ValueError("Require frozen12independent development groups")
    memory_limit = int(Path("/sys/fs/cgroup/memory.max").read_text())
    current = int(Path("/sys/fs/cgroup/memory.current").read_text())
    if memory_limit - current - spec["bytes"] < 2 * 1024**3:
        raise RuntimeError("Insufficient declared memory headroom")
    if shutil.disk_usage(MODEL.parent).free < spec["bytes"] + 64 * 1024**2:
        raise RuntimeError("Insufficient task-owned shared memory space")
    write_progress({"stage": "model_acquisition", "pid": os.getpid()})
    start = time.monotonic()
    digest = hashlib.sha256()
    size = 0
    url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{spec['file']}"
    with urllib.request.urlopen(url, timeout=60) as response, MODEL.open("xb") as out:
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > spec["bytes"]:
                raise ValueError("Checkpoint size exceeded pinned bound")
            digest.update(chunk)
            out.write(chunk)
            if size % (256 * 1024**2) == 0:
                write_progress(
                    {
                        "stage": "model_acquisition",
                        "bytes": size,
                        "expected_bytes": spec["bytes"],
                        "seconds": time.monotonic() - start,
                    }
                )
    if size != spec["bytes"] or digest.hexdigest() != spec["sha256"]:
        raise ValueError("Checkpoint integrity mismatch")
    acquisition = time.monotonic() - start
    socket.socket.connect = blocked_connect
    if importlib.metadata.version("llama-cpp-python") != "0.3.19":
        raise ValueError("Runtime differs from frozen config")
    write_progress(
        {"stage": "model_loading", "bytes_verified": size, "acquisition_seconds": acquisition}
    )
    model = Llama(
        model_path=str(MODEL),
        n_ctx=4096,
        n_batch=16,
        n_threads=2,
        n_threads_batch=2,
        n_gpu_layers=0,
        use_mmap=True,
        verbose=False,
        seed=1989,
    )
    try:
        ready_deadline = time.monotonic() + 60
        ready = chat_with_stopping(
            model,
            StoppingCriteriaList([lambda ids, scores: time.monotonic() >= ready_deadline]),
            messages=[
                {
                    "role": "user",
                    "content": "Return exactly a JSON object with a boolean field named ready.",
                }
            ],
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {"ready": {"type": "boolean"}},
                    "required": ["ready"],
                    "additionalProperties": False,
                },
            },
            max_tokens=16,
            temperature=0,
            seed=1989,
        )
        probe = json.loads(ready["choices"][0]["message"]["content"])
        if set(probe) != {"ready"} or type(probe["ready"]) is not bool:
            raise RuntimeError("Structured native readiness probe failed before dataset scoring")
        (PRIVATE / "readiness.json").write_text(
            json.dumps(
                {
                    "native_chat_and_stopping_passed": True,
                    "usage": ready.get("usage", {}),
                    "unscored_synthetic_probe": True,
                }
            )
            + "\n"
        )
        tasks = [
            (mode, row) for mode in ["direct", "reasoning-first"] for row in inputs["questions"]
        ]
        random.Random(1989).shuffle(tasks)
        deadline = time.monotonic() + 2100
        records = []
        calls = 0
        for mode, row in tasks:
            base = {k: row[k] for k in ["id", "group_id", "expected"]} | {"mode": mode}
            if time.monotonic() + 45 > deadline:
                records.append(
                    base
                    | {
                        "candidate": "MAYBE",
                        "failure": "DEFERRED_TIME_LIMIT",
                        "evidence_indices": [],
                        "latency_seconds": None,
                        "request_attempted": False,
                    }
                )
                continue
            call_events = []

            def generate(*, phase, task_mode=mode, task_events=call_events, **kwargs):
                nonlocal calls
                calls += 1
                begin = time.monotonic()
                phase_deadline = min(deadline, begin + (180 if phase == "analysis" else 45))
                stopped = [False]
                last = [0.0]

                def stop(ids, logits):
                    now = time.monotonic()
                    if now - last[0] >= 20:
                        write_progress(
                            {
                                "stage": "generation",
                                "mode": task_mode,
                                "phase": phase,
                                "completed_slots": len(records),
                                "token_sequence_length": len(ids),
                                "phase_seconds": now - begin,
                            }
                        )
                        last[0] = now
                    if now >= phase_deadline:
                        stopped[0] = True
                        return True
                    return False

                answer = chat_with_stopping(model, StoppingCriteriaList([stop]), **kwargs)
                answer["time_stopped"] = stopped[0]
                task_events.append(
                    {
                        "phase": phase,
                        "time_stopped": stopped[0],
                        "seconds": time.monotonic() - begin,
                        "usage": answer.get("usage", {}),
                    }
                )
                return answer

            draft_path = PRIVATE / (row["id"].replace(":", "_") + "-draft.txt")
            try:
                result = predict(
                    generate,
                    row["question"],
                    row["evidence"],
                    mode=mode,
                    observer=lambda text, target=draft_path: target.write_text(text),
                )
                item = base | result | {"request_attempted": True, "calls": call_events}
            except Exception as exc:  # noqa: BLE001 -- retain failed slots without logging prompts/URLs
                item = base | {
                    "candidate": "MAYBE",
                    "failure": type(exc).__name__,
                    "evidence_indices": [],
                    "latency_seconds": None,
                    "request_attempted": True,
                    "calls": call_events,
                }
            records.append(item)
            (PRIVATE / "partial.json").write_text(
                json.dumps({"completed": False, "records": records}) + "\n"
            )
            write_progress(
                {
                    "stage": "scoring",
                    "completed_slots": len(records),
                    "planned_slots": 24,
                    "model_calls": calls,
                }
            )
            print(
                json.dumps(
                    {"completed_slots": len(records), "mode": mode, "failure": item["failure"]}
                ),
                flush=True,
            )
        result = {
            "completed": True,
            "records": records,
            "model": spec,
            "runtime": importlib.metadata.version("llama-cpp-python"),
            "model_calls": calls,
            "additional_unscored_readiness_calls": 1,
            "acquisition_seconds": acquisition,
            "total_seconds": time.monotonic() - start,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "inference_network": "disabled",
            "production_qualified": False,
            "new_cloud_spend_usd": 0,
        }
        (PRIVATE / "result.json").write_text(json.dumps(result) + "\n")
        write_progress({"stage": "completed", "slots": len(records), "model_calls": calls})
    finally:
        model.close()


if __name__ == "__main__":
    main()
