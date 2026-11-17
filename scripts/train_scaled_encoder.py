"""One time-limited additional training epoch, suitable for the isolated AWS CPU worker."""

import hashlib
import importlib.metadata
import json
import random
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import RobertaForSequenceClassification


def train(root, *, progress=None):
    random.seed(1935)
    np.random.seed(1935)
    torch.manual_seed(1935)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    base = root / "initial"
    output = root / "trained"
    if output.exists():
        raise FileExistsError("Never overwrite a checkpoint")
    expected = json.loads((root / "input-manifest.json").read_text())
    for name, sha in expected["initial_sha256"].items():
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Initial model integrity failure")
    inputs = root / "training-token-inputs.jsonl"
    if hashlib.sha256(inputs.read_bytes()).hexdigest() != expected["token_inputs_sha256"]:
        raise ValueError("Training data integrity failure")
    rows = [json.loads(line) for line in inputs.read_text().splitlines()]
    random.shuffle(rows)
    model = RobertaForSequenceClassification.from_pretrained(
        base, local_files_only=True, use_safetensors=True, attn_implementation="eager"
    )
    model.config.use_cache = False
    initial = model.classifier.out_proj.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    losses = []
    started = time.perf_counter()
    completed = True
    model.train()
    for offset in range(0, len(rows), 8):
        if time.perf_counter() - started >= 1800:
            completed = False
            break
        batch = rows[offset : offset + 8]
        length = max(len(r["ids"]) for r in batch)
        ids = torch.tensor([r["ids"] + [1] * (length - len(r["ids"])) for r in batch])
        mask = torch.tensor([r["mask"] + [0] * (length - len(r["mask"])) for r in batch])
        types = torch.tensor([r["types"] + [0] * (length - len(r["types"])) for r in batch])
        labels = torch.tensor([r["label"] for r in batch])
        optimizer.zero_grad(set_to_none=True)
        result = model(input_ids=ids, attention_mask=mask, token_type_ids=types, labels=labels)
        result.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        losses.append(float(result.loss.detach()))
        if len(losses) % 25 == 0:
            item = {
                "steps": len(losses),
                "rows": min(offset + 8, len(rows)),
                "mean_loss_last25": sum(losses[-25:]) / 25,
                "training_seconds": time.perf_counter() - started,
            }
            print(json.dumps(item), flush=True)
            if progress:
                progress(item)
    model.eval()
    model.save_pretrained(output, safe_serialization=True)
    shutil.copyfile(base / "tokenizer.json", output / "tokenizer.json")
    training_seconds = time.perf_counter() - started
    record = {
        "seed": 1935,
        "completed": completed,
        "epochs": 1 if completed else None,
        "rows": len(rows),
        "steps": len(losses),
        "optimizer": "AdamW reset",
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "batch_size": 8,
        "cpu_threads": 8,
        "max_length": 256,
        "training_seconds": training_seconds,
        "losses": losses,
        "classifier_update_l2": float(
            torch.linalg.vector_norm(model.classifier.out_proj.weight.detach() - initial)
        ),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["torch", "transformers", "numpy", "safetensors", "onnx", "onnxruntime"]
        },
        "initial_sha256": expected["initial_sha256"],
        "checkpoint_sha256": hashlib.sha256(
            (output / "model.safetensors").read_bytes()
        ).hexdigest(),
        "production_qualified": False,
        "cloud_invoiced_cost_usd": None,
        "note": "One fixed additional epoch on new independent real revision supervision plus replay. No calibration/test labels on worker. No validation-based checkpoint selection.",
    }
    (output / "training.json").write_text(json.dumps(record, indent=2) + "\n")
    if completed:

        class Export(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, input_ids, attention_mask, token_type_ids):
                return self.inner(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                ).logits

        wrapper = Export(model).eval()
        p = output / "onnx/model.onnx"
        p.parent.mkdir()
        sample = torch.tensor([[0, 10, 11, 12, 2]])
        names = ["input_ids", "attention_mask", "token_type_ids"]
        torch.onnx.export(
            wrapper,
            (sample, torch.ones_like(sample), torch.zeros_like(sample)),
            str(p),
            input_names=names,
            output_names=["logits"],
            dynamic_axes={k: {0: "batch", 1: "sequence"} for k in names} | {"logits": {0: "batch"}},
            opset_version=17,
            external_data=False,
            dynamo=False,
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(p), sess_options=options, providers=["CPUExecutionProvider"]
        )
        checks = []
        for batch, length in [(1, 8), (2, 64), (1, 256), (2, 256)]:
            ids = torch.randint(5, model.config.vocab_size, (batch, length))
            mask = torch.ones_like(ids)
            types = torch.zeros_like(ids)
            with torch.inference_mode():
                ref = wrapper(ids, mask, types).numpy()
            actual = session.run(None, dict(zip(names, [x.numpy() for x in (ids, mask, types)])))[0]
            error = float(np.max(np.abs(actual - ref)))
            if error > 1e-4 or not np.array_equal(actual.argmax(1), ref.argmax(1)):
                raise ValueError("Export parity failed")
            checks.append({"batch": batch, "length": length, "max_absolute_error": error})
        manifest = {
            "model_id": "houndoracle/minilm-vitc-scaled-1935",
            "revision": record["checkpoint_sha256"],
            "declared_license": "Private research checkpoint; redistribution requires upstream model/data license review",
            "max_length": 256,
            "files": {},
        }
        for name in ["config.json", "tokenizer.json", "onnx/model.onnx"]:
            file = output / name
            manifest["files"][name] = {
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                "bytes": file.stat().st_size,
            }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (output / "parity.json").write_text(
            json.dumps({"passed": True, "checks": checks}, indent=2) + "\n"
        )
    with tarfile.open(root / "trained.tar.gz", "w:gz", compresslevel=1) as archive:
        archive.add(output, arcname="trained")
    return record


def main():
    import boto3
    from botocore.config import Config

    root = Path("/opt/houndoracle")
    s3 = boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(
            retries={"total_max_attempts": 2, "mode": "standard"},
            connect_timeout=10,
            read_timeout=60,
        ),
    )
    bucket = "houndoracle-research-751446514671-20261116"

    def progress(item):
        s3.put_object(
            Bucket=bucket,
            Key="training19/output/progress.json",
            Body=json.dumps(item).encode(),
            ContentType="application/json",
        )

    try:
        record = train(root, progress=progress)
        s3.upload_file(str(root / "trained.tar.gz"), bucket, "training19/output/trained.tar.gz")
        s3.put_object(
            Bucket=bucket,
            Key="training19/output/training.json",
            Body=json.dumps(record).encode(),
            ContentType="application/json",
        )
        progress(
            {
                "state": "COMPLETED" if record["completed"] else "INCOMPLETE",
                "steps": record["steps"],
                "training_seconds": record["training_seconds"],
            }
        )
    except BaseException as exc:
        progress({"state": "FAILED", "exception_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
