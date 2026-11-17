"""One bounded full-precision graph-fusion experiment; no fitting or quantization."""

import hashlib
import json
import shutil
import time
from pathlib import Path

from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.optimizer import optimize_model


def main():
    source = Path("/workspace/houndoracle-models/minilm-vitc-fever-1921")
    target = source.with_name(source.name + "-fused")
    if target.exists():
        raise FileExistsError("Graph experiment already exists")
    (target / "onnx").mkdir(parents=True)
    for name in ["config.json", "tokenizer.json"]:
        shutil.copyfile(source / name, target / name)
    options = FusionOptions("bert")
    options.enable_gelu_approximation = False
    options.enable_embed_layer_norm = False
    start = time.perf_counter()
    model = optimize_model(
        str(source / "onnx/model.onnx"),
        model_type="bert",
        num_heads=12,
        hidden_size=768,
        optimization_options=options,
        opt_level=0,
        use_gpu=False,
    )
    stats = model.get_fused_operator_statistics()
    model.save_model_to_file(str(target / "onnx/model.onnx"))
    manifest = {
        "model_id": "houndoracle/minilm-vitc-fever-1921-fused",
        "revision": hashlib.sha256((target / "onnx/model.onnx").read_bytes()).hexdigest(),
        "max_length": 256,
        "declared_license": "Local research; source/model redistribution compatibility review required",
        "files": {},
        "optimization": {
            "runtime": "1.32.0",
            "model_type": "bert-compatible Roberta",
            "heads": 12,
            "hidden_size": 768,
            "ort_preoptimization_level": 0,
            "gelu_approximation": False,
            "embedding_layernorm_fusion": False,
            "source_sha256": hashlib.sha256((source / "onnx/model.onnx").read_bytes()).hexdigest(),
            "fused_operator_statistics": stats,
            "seconds": time.perf_counter() - start,
        },
    }
    for name in ["config.json", "tokenizer.json", "onnx/model.onnx"]:
        path = target / name
        manifest["files"][name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    Path("research/experiments/campaign-13/fused-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest["optimization"], indent=2))


if __name__ == "__main__":
    main()
