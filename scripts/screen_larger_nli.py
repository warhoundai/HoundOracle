"""Pinned ONNX CPU screen; never opens fresh risk/test or executes remote model code."""

import hashlib
import importlib.metadata
import json
import resource
import time
import urllib.request
from pathlib import Path


def main():
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    root = Path("/opt/hound28")
    inputs = json.loads((root / "inputs.json").read_text())
    model_dir = root / "model"
    model_dir.mkdir(exist_ok=True)
    start = time.monotonic()
    downloaded = {}
    for name, seal in inputs["files"].items():
        target = model_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{inputs['model_id']}/resolve/{inputs['revision']}/{name}"
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(url, timeout=30) as response, target.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > seal["bytes"]:
                    raise ValueError("Model size exceeded reviewed seal")
                digest.update(chunk)
                out.write(chunk)
        if size != seal["bytes"] or digest.hexdigest() != seal["sha256"]:
            raise ValueError("Model integrity mismatch")
        downloaded[name] = {"bytes": size, "sha256": digest.hexdigest()}
        print(json.dumps({"downloaded": name, "bytes": size}), flush=True)
    acquisition = time.monotonic() - start
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    full_tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=256)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    then = time.monotonic()
    session = ort.InferenceSession(
        str(model_dir / "onnx/model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    load_time = time.monotonic() - then
    config = json.loads((model_dir / "config.json").read_text())
    labels = {int(k): v.lower() for k, v in config["id2label"].items()}
    verdict = {"entailment": "YES", "neutral": "MAYBE", "contradiction": "NO"}
    if set(labels.values()) != set(verdict):
        raise ValueError("Unreviewed model labels")
    records = []
    for i, row in enumerate(inputs["rows"]):
        then = time.monotonic()
        passage = "\n".join(e["text"] for e in row["evidence"])
        record = {k: row[k] for k in ["id", "group_id", "expected", "baseline_candidate"]}
        if not passage:
            record.update(candidate="MAYBE", raw_score=None, truncated=False, model_input_tokens=0)
        else:
            full = full_tokenizer.encode(passage, row["question"])
            enc = tokenizer.encode(passage, row["question"])
            values = {
                "input_ids": np.array([enc.ids], dtype=np.int64),
                "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
                "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
            }
            logits = np.asarray(
                session.run(None, {x.name: values[x.name] for x in session.get_inputs()})[0][0],
                dtype=float,
            )
            p = np.exp(logits - logits.max())
            p /= p.sum()
            pick = int(p.argmax())
            record.update(
                candidate=verdict[labels[pick]],
                raw_score=float(p[pick]),
                logits_by_label={labels[j]: float(v) for j, v in enumerate(logits)},
                truncated=len(full.ids) > 256,
                model_input_tokens=len(enc.ids),
            )
        record.update(latency_seconds=time.monotonic() - then, citation_supported=None)
        records.append(record)
        if (i + 1) % 25 == 0:
            print(json.dumps({"scored": i + 1, "seconds": time.monotonic() - start}), flush=True)
    n = len(records)
    b = sum(r["baseline_candidate"] == r["expected"] for r in records) / n
    c = sum(r["candidate"] == r["expected"] for r in records) / n
    report = {
        "model_id": inputs["model_id"],
        "revision": inputs["revision"],
        "count": n,
        "baseline_raw_accuracy": b,
        "candidate_raw_accuracy": c,
        "raw_accuracy_delta": c - b,
        "screen_passed": c - b >= 0.02,
        "acquisition_seconds": acquisition,
        "load_seconds": load_time,
        "total_seconds": time.monotonic() - start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "files": downloaded,
        "versions": {
            p: importlib.metadata.version(p) for p in ["onnxruntime", "tokenizers", "numpy"]
        },
        "records": records,
        "scope": "Reused supplied-evidence development screen, not live retrieval, calibrated correctness or99% qualification. Pretraining overlap unknown. No public source text/weights in output.",
    }
    (root / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
