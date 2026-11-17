"""One predeclared FP32 parameter average; preserve parent checkpoints and verify export."""

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from safetensors.torch import load_file
from transformers import RobertaForSequenceClassification


def main():
    target = Path("/workspace/houndoracle-models/minilm-vitc-average-1932")
    if target.exists():
        raise FileExistsError("Do not overwrite research weights")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1932)
    parents = [
        Path("/workspace/houndoracle-models/minilm-vitc-fever-" + str(n)) for n in (1920, 1921)
    ]
    source = [load_file(p / "model.safetensors") for p in parents]
    if source[0].keys() != source[1].keys():
        raise ValueError("Different parameter schemas")
    started = time.perf_counter()
    averaged = {}
    for key, left in source[0].items():
        right = source[1][key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError("Incompatible parameter tensors")
        if torch.is_floating_point(left):
            averaged[key] = (left.float() + right.float()) * 0.5
        else:
            if not torch.equal(left, right):
                raise ValueError("Nonfloating tensors differ")
            averaged[key] = left
    model = RobertaForSequenceClassification.from_pretrained(
        parents[1], local_files_only=True, use_safetensors=True, attn_implementation="eager"
    )
    model.load_state_dict(averaged, strict=True)
    model.eval()
    model.save_pretrained(target, safe_serialization=True)
    shutil.copyfile(parents[1] / "tokenizer.json", target / "tokenizer.json")
    del source, averaged

    class Export(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.inner(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            ).logits

    wrapper = Export(model).eval()
    destination = target / "onnx/model.onnx"
    destination.parent.mkdir()
    sample = torch.tensor([[0, 10, 11, 12, 2]], dtype=torch.long)
    torch.onnx.export(
        wrapper,
        (sample, torch.ones_like(sample), torch.zeros_like(sample)),
        str(destination),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            k: {0: "batch", 1: "sequence"}
            for k in ["input_ids", "attention_mask", "token_type_ids"]
        }
        | {"logits": {0: "batch"}},
        opset_version=17,
        external_data=False,
        dynamo=False,
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(destination), sess_options=options, providers=["CPUExecutionProvider"]
    )
    checks = []
    for batch, length in [(1, 8), (2, 64), (1, 256), (2, 256)]:
        ids = torch.randint(5, model.config.vocab_size, (batch, length))
        mask, types = torch.ones_like(ids), torch.zeros_like(ids)
        with torch.inference_mode():
            expected = wrapper(ids, mask, types).numpy()
        actual = session.run(
            None,
            dict(
                zip(
                    ["input_ids", "attention_mask", "token_type_ids"],
                    [x.numpy() for x in (ids, mask, types)],
                )
            ),
        )[0]
        error = float(np.max(np.abs(actual - expected)))
        if error > 1e-4 or not np.array_equal(actual.argmax(1), expected.argmax(1)):
            raise ValueError("Export parity failed")
        checks.append({"batch": batch, "length": length, "max_error": error})
    manifest = {
        "model_id": "houndoracle/minilm-vitc-average-1932",
        "revision": hashlib.sha256((target / "model.safetensors").read_bytes()).hexdigest(),
        "declared_license": "Local research; upstream model and dataset redistribution rights require review",
        "max_length": 256,
        "files": {},
    }
    for name in ["config.json", "tokenizer.json", "onnx/model.onnx"]:
        p = target / name
        manifest["files"][name] = {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    public = Path("research/experiments/campaign-16")
    (public / "averaged-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    record = {
        "parents": {
            p.name: hashlib.sha256((p / "model.safetensors").read_bytes()).hexdigest()
            for p in parents
        },
        "weights": [0.5, 0.5],
        "parity_checks": checks,
        "seconds": time.perf_counter() - started,
        "gradient_updates": 0,
        "new_cloud_cost_usd": 0,
        "production_qualified": False,
    }
    (public / "parameter-average.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
