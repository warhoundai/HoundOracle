"""One fixed CE-adapter continuation on additional supervised training components."""

import hashlib
import json
from pathlib import Path

import torch
import train_contrastive_adapters as trainer


def main():
    private = Path("/workspace/houndoracle-data/campaign-38")
    public = Path("research/experiments/campaign-38")
    protocol = json.loads((public / "protocol.json").read_text())
    if (
        hashlib.file_digest((private / "inputs.json").open("rb"), "sha256").hexdigest()
        != protocol["inputs_sha256"]
    ):
        raise ValueError("Frozen continuation inputs changed")
    if (
        hashlib.file_digest((trainer.BASE / "model.safetensors").open("rb"), "sha256").hexdigest()
        != protocol["base_checkpoint_sha256"]
    ):
        raise ValueError("Base checkpoint changed")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    references = json.loads(
        Path("/workspace/houndoracle-data/campaign-35/ce-control/result.json").read_text()
    )["validation_scores"][:3]
    trainer.PRIVATE = private
    complete = trainer.train(
        "ce-continuation",
        0,
        json.loads((private / "inputs.json").read_text()),
        protocol,
        initial_adapter=Path(
            "/workspace/houndoracle-data/campaign-35/ce-control/adapter.safetensors"
        ),
        parity_references=references,
        maximum_training_seconds=1200,
    )
    (private / "completed.json").write_text(json.dumps({"completed": complete}) + "\n")


if __name__ == "__main__":
    main()
