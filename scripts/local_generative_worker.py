"""One bounded local-model experiment; upload observations, never source text."""

import hashlib
import importlib.metadata
import json
import resource
import time
import urllib.request
from pathlib import Path

from houndoracle.experiments.generative import LocalGenerativeVerifier


def main():
    import boto3
    from botocore.config import Config
    from llama_cpp import Llama

    root = Path("/opt/hound32")
    data = json.loads((root / "inputs.json").read_text())
    s3 = boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(
            connect_timeout=5,
            read_timeout=30,
            retries={"total_max_attempts": 2, "mode": "standard"},
        ),
    )
    bucket, prefix = "houndoracle-research-751446514671-20261116", "qwen32/output/"

    def save(name, value):
        s3.put_object(Bucket=bucket, Key=prefix + name, Body=json.dumps(value).encode())

    start = time.monotonic()
    spec = data["model"]
    file = root / "model.gguf"
    url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{spec['file']}"
    digest, size = hashlib.sha256(), 0
    with urllib.request.urlopen(url, timeout=60) as response, file.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > spec["bytes"]:
                raise ValueError("Model exceeds reviewed size")
            digest.update(chunk)
            out.write(chunk)
    if size != spec["bytes"] or digest.hexdigest() != spec["sha256"]:
        raise ValueError("Model integrity mismatch")
    save("progress.json", {"stage": "model_verified", "download_seconds": time.monotonic() - start})
    acquisition = time.monotonic() - start
    model = Llama(
        model_path=str(file),
        n_ctx=4096,
        n_threads=8,
        n_threads_batch=8,
        n_batch=256,
        seed=1970,
        verbose=False,
    )
    predictor = LocalGenerativeVerifier(
        model,
        config=spec
        | {
            "context": 4096,
            "threads": 8,
            "runtime": importlib.metadata.version("llama-cpp-python"),
        },
    )
    rows = []
    # Full initial dataset selection and baseline existed before any model answer.
    for i, row in enumerate(data["questions"]):
        pred = predictor.predict(row["question"], row["evidence"])
        item = {k: row[k] for k in ["id", "group_id", "expected"]} | pred
        item["baseline_candidate"] = data["baseline"][i]["candidate"]
        rows.append(item)
        save("partial.json", {"completed": False, "rows": rows})
        save(
            "progress.json",
            {"stage": "scoring", "rows": len(rows), "seconds": time.monotonic() - start},
        )
        if len(rows) <= 3 and all(x.get("failure") for x in rows) and len(rows) == 3:
            raise ValueError("Three initial invalid generations; stop compatibility failure")
    result = {
        "completed": True,
        "rows": rows,
        "model": spec,
        "runtime": predictor.config,
        "acquisition_seconds": acquisition,
        "total_seconds": time.monotonic() - start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "production_qualified": False,
    }
    save("result.json", result)
    save("progress.json", {"stage": "completed", "rows": len(rows)})


if __name__ == "__main__":
    main()
