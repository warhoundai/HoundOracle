"""Bounded private CPU experiment; never reads validation/calibration/locked cohorts."""

import gc
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import save_file

from houndoracle import DocumentInput, Oracle
from houndoracle.experiments.low_rank import attach, trainable_state
from houndoracle.general import GeneralVerifier
from houndoracle.local_semantics import LocalPairEvaluator


def main():
    root = Path(os.environ.get("HOUNDORACLE_EXPERIMENT_ROOT", "/opt/hound46"))
    payload = json.loads((root / "payload.json").read_text())
    protocol = payload["protocol"]
    training = payload["training"]
    development = payload["development"]
    if len(training) != 4096 or len(development) != 500:
        raise ValueError("Fixed pilot dimensions required")
    if len({r["group_id"] for r in training + development}) != 4596:
        raise ValueError("Source component overlap")
    if any(r["label_origin"] != "human_dataset" for r in training + development):
        raise ValueError("Human annotation source required")
    torch.set_num_interop_threads(1)
    torch.set_num_threads(8)
    random.seed(2005)
    torch.manual_seed(2005)
    started = time.monotonic()
    out = root / "output"
    out.mkdir(exist_ok=True)

    def progress(stage, **kw):
        record = {"stage": stage, **kw}
        (out / "progress.json").write_text(json.dumps(record))
        print(json.dumps(record), flush=True)

    def evaluate(evaluator, label):
        observations = []
        for i, r in enumerate(development):
            oracle = Oracle(general_verifier=GeneralVerifier(evaluator))
            try:
                for e in r["evidence"]:
                    now = time.time()
                    oracle.store.ingest(
                        DocumentInput(
                            url=e["url"],
                            text=e["text"],
                            origin="public-human-annotation-snapshot",
                            observed_at=now,
                            expires_at=now + 3600,
                            reviewed=True,
                            retrieval_kind="source_document",
                        )
                    )
                result = oracle.verify(r["question"], max_seconds=120)
                trace = result.diagnostics
                observations.append(
                    {k: r[k] for k in ["id", "group_id", "expected"]}
                    | {
                        "candidate": trace.candidate_decision.value if trace else "MAYBE",
                        "decision": result.decision.value,
                        "raw_score": trace.raw_score if trace else None,
                        "reason": result.reason,
                        "policy_id": result.versions["policy"],
                        "latency_seconds": result.usage["latency_seconds"],
                        "confidence": result.confidence,
                        "failure": result.reason
                        if result.reason in ["TIME_EXHAUSTED", "SEMANTIC_EVALUATION_FAILED"]
                        else None,
                    }
                )
            finally:
                oracle.store.close()
            if (i + 1) % 100 == 0:
                progress(label, scored=i + 1, total=500)
        return observations

    evaluator = LocalPairEvaluator(
        root / "base", expected_weights_sha256=protocol["base_sha256"], threads=8
    )
    progress("baseline")
    baseline = evaluate(evaluator, "baseline")
    model = evaluator.model
    tokenizer = evaluator.tokenizer
    attach(model, rank=8, last_layers=4)
    labels = {v: k for k, v in evaluator.labels.items()}
    mapping = {
        "NO": labels["contradiction"],
        "YES": labels["entailment"],
        "MAYBE": labels["neutral"],
    }
    counts = Counter(r["expected"] for r in training)
    class_weights = torch.tensor(
        [
            len(training)
            / (
                3
                * counts[
                    {"contradiction": "NO", "entailment": "YES", "neutral": "MAYBE"}[
                        evaluator.labels[i]
                    ]
                ]
            )
            for i in range(3)
        ]
    )
    adapted = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("classifier.")
    ]
    classifier = [
        p for n, p in model.named_parameters() if p.requires_grad and n.startswith("classifier.")
    ]
    optimizer = torch.optim.AdamW(
        [{"params": adapted, "lr": 5e-5}, {"params": classifier, "lr": 5e-6}], weight_decay=0.01
    )
    shuffled = list(training)
    random.Random(2005).shuffle(shuffled)
    curve = []
    deadline = time.monotonic() + 1200
    steps = 0
    model.train()
    train_start = time.monotonic()
    for start in range(0, len(shuffled), 8):
        if time.monotonic() >= deadline:
            break
        rows = shuffled[start : start + 8]
        tokens = tokenizer(
            ["\n".join(e["text"] for e in r["evidence"])[:800] for r in rows],
            [r["question"] for r in rows],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        target = torch.tensor([mapping[r["expected"]] for r in rows])
        optimizer.zero_grad(set_to_none=True)
        logits = model(**tokens).logits
        loss = torch.nn.functional.cross_entropy(logits, target, weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1)
        optimizer.step()
        steps += 1
        curve.append(
            {
                "step": steps,
                "loss": float(loss.detach()),
                "elapsed_seconds": time.monotonic() - train_start,
            }
        )
        if steps % 50 == 0:
            progress("training", steps=steps, expected_steps=512, loss=curve[-1]["loss"])
    train_seconds = time.monotonic() - train_start
    adapter = out / "adapter.safetensors"
    save_file(trainable_state(model), str(adapter))
    adapter_sha = hashlib.file_digest(adapter.open("rb"), "sha256").hexdigest()
    del optimizer, model, evaluator
    gc.collect()
    candidate = LocalPairEvaluator(
        root / "base",
        expected_weights_sha256=protocol["base_sha256"],
        threads=8,
        adapter_path=adapter,
        expected_adapter_sha256=adapter_sha,
    )
    progress("candidate")
    scores = evaluate(candidate, "candidate")
    result = {
        "baseline": baseline,
        "candidate": scores,
        "training_steps": steps,
        "training_complete": steps == math.ceil(len(training) / 8),
        "training_seconds": train_seconds,
        "curve": curve,
        "adapter_sha256": adapter_sha,
        "model_configuration": candidate.config,
        "class_counts": dict(counts),
        "class_weights": class_weights.tolist(),
        "total_seconds": time.monotonic() - started,
        "production_qualified": False,
    }
    (out / "result.json").write_text(json.dumps(result))
    progress("completed", training_steps=steps)


if __name__ == "__main__":
    main()
