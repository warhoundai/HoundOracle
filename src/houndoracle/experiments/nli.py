"""Pinned, CPU-only ONNX NLI challenger. No production routing or remote model code."""

import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

MODEL_ID = "cross-encoder/nli-MiniLM2-L6-H768"
REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
REVIEWED_MODELS = {
    MODEL_ID: REVISION,
    "cross-encoder/nli-deberta-v3-small": "fa2804872c3b4bd748f38c0185cc85775361e735",
}
FILES = {"fp32": "onnx/model.onnx", "int8": "onnx/model_quint8_avx2.onnx"}


def download(directory):
    from huggingface_hub import hf_hub_download

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = ["config.json", "tokenizer.json", "README.md", *FILES.values()]
    manifest = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "declared_license": "Apache-2.0",
        "pretraining_datasets": ["SNLI", "MultiNLI"],
        "files": {},
        "rights_note": "Model card reviewed. Training-data and base-model rights remain separate; no public model redistribution in this campaign.",
    }
    for filename in files:
        path = Path(
            hf_hub_download(MODEL_ID, filename, revision=REVISION, local_dir=directory, token=False)
        )
        if path.stat().st_size > 400_000_000:
            raise ValueError("Model file exceeds size budget")
        manifest["files"][filename] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class OnnxNLI:
    def __init__(
        self, directory, *, variant="int8", threads=1, reviewed_manifest=None, max_length=None
    ):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.directory = Path(directory)
        manifest = json.loads((self.directory / "manifest.json").read_text())
        if reviewed_manifest is not None:
            if manifest != reviewed_manifest:
                raise ValueError(
                    "Locally trained model differs from the explicit reviewed manifest"
                )
        elif REVIEWED_MODELS.get(manifest["model_id"]) != manifest["revision"]:
            raise ValueError("Model revision has not been reviewed")
        manifest_limit = manifest.get("max_length", 512)
        if type(manifest_limit) is not int or not 32 <= manifest_limit <= 512:
            raise ValueError("Unsupported model token envelope")
        self.max_length = manifest_limit if max_length is None else max_length
        if type(self.max_length) is not int or not 32 <= self.max_length <= manifest_limit:
            raise ValueError("Requested token envelope exceeds the reviewed model limit")
        for filename in ["config.json", "tokenizer.json", FILES[variant]]:
            if (
                hashlib.sha256((self.directory / filename).read_bytes()).hexdigest()
                != manifest["files"][filename]["sha256"]
            ):
                raise ValueError("Model integrity check failed")
        self.tokenizer = Tokenizer.from_file(str(self.directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.max_length)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(self.directory / FILES[variant]),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        config = json.loads((self.directory / "config.json").read_text())
        self.labels = [config["id2label"][str(i)] for i in range(3)]
        self.config = {
            "model_id": manifest["model_id"],
            "revision": manifest["revision"],
            "variant": variant,
            "threads": threads,
            "max_length": self.max_length,
            "runtime": importlib.metadata.version("onnxruntime"),
            "tokenizers": importlib.metadata.version("tokenizers"),
            "file_sha256": manifest["files"][FILES[variant]]["sha256"],
        }

    def predict(self, question, evidence):
        import numpy as np

        started = time.perf_counter()
        # Fixed original FEVER evidence order; use identical concatenation for all compared systems.
        passage = "\n".join(e["text"] for e in evidence)
        if not passage:
            return {
                "candidate": "MAYBE",
                "raw_score": None,
                "logits": None,
                "model_seconds": time.perf_counter() - started,
                "truncated": False,
            }
        encoding = self.tokenizer.encode(passage, question)
        values = {
            "input_ids": np.array([encoding.ids], dtype=np.int64),
            "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([encoding.type_ids], dtype=np.int64),
        }
        feed = {x.name: values[x.name] for x in self.session.get_inputs()}
        logits = np.asarray(self.session.run(None, feed)[0][0], dtype=float)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        index = int(probs.argmax())
        candidate = {"entailment": "YES", "contradiction": "NO", "neutral": "MAYBE"}[
            self.labels[index]
        ]
        return {
            "candidate": candidate,
            "raw_score": float(probs[index]),
            "logits": logits.tolist(),
            "probabilities": probs.tolist(),
            "model_seconds": time.perf_counter() - started,
            "truncated": len(encoding.ids) >= self.max_length,
            "token_count": len(encoding.ids),
        }
