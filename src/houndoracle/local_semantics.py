"""Pinned local pair evaluator. No automatic downloads or model-generated code."""

import hashlib
import importlib.metadata
import json
import threading
from pathlib import Path


class LocalPairEvaluator:
    def __init__(
        self,
        directory,
        *,
        expected_weights_sha256,
        max_length=256,
        threads=2,
        adapter_path=None,
        expected_adapter_sha256=None,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if max_length != 256 or not 1 <= threads <= 8:
            raise ValueError("Unsupported semantic inference configuration")
        directory = Path(directory)
        weights = directory / "model.safetensors"
        actual = hashlib.file_digest(weights.open("rb"), "sha256").hexdigest()
        if actual != expected_weights_sha256:
            raise ValueError("Local checkpoint integrity mismatch")
        torch.set_num_threads(threads)
        self.lock = threading.RLock()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            directory,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )
        self.labels = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        if set(self.labels.values()) != {"entailment", "contradiction", "neutral"}:
            raise ValueError("Expected explicit three-way NLI labels")
        adapter_sha = None
        if adapter_path is not None:
            from safetensors.torch import load_file

            from .experiments.low_rank import attach, restore_trainable

            adapter_path = Path(adapter_path)
            adapter_sha = hashlib.file_digest(adapter_path.open("rb"), "sha256").hexdigest()
            if adapter_sha != expected_adapter_sha256:
                raise ValueError("Local adapter integrity mismatch")
            attach(self.model, rank=8, last_layers=4)
            restore_trainable(self.model, load_file(str(adapter_path)))
            self.model.eval()
        elif expected_adapter_sha256 is not None:
            raise ValueError("Adapter hash requires adapter weights")
        assets = {
            p.name: hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
            for p in directory.iterdir()
            if p.suffix == ".json"
        }
        self.config = {
            "weights_sha256": actual,
            "adapter_sha256": adapter_sha,
            "assets": assets,
            "max_length": max_length,
            "dtype": "float32",
            "threads": threads,
            "packing": "passage-exact-proposition-v1",
            "torch_version": importlib.metadata.version("torch"),
            "transformers_version": importlib.metadata.version("transformers"),
        }
        self.version = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        self.max_length = max_length

    def predict(self, question, passage):
        import torch

        with self.lock, torch.inference_mode():
            full = self.tokenizer(passage, question, truncation=False)["input_ids"]
            tokens = self.tokenizer(
                passage, question, truncation=True, max_length=self.max_length, return_tensors="pt"
            )
            probabilities = torch.softmax(self.model(**tokens).logits[0], dim=-1).tolist()
            choice = max(range(len(probabilities)), key=probabilities.__getitem__)
            labels = {"entailment": "YES", "contradiction": "NO", "neutral": "MAYBE"}
            return {
                "candidate": labels[self.labels[choice]],
                "raw_score": probabilities[choice],
                "truncated": len(full) > self.max_length,
                "probabilities": {labels[self.labels[i]]: p for i, p in enumerate(probabilities)},
            }
