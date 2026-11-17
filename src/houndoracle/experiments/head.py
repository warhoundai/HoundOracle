"""Safe JSON checkpoints for supervised aggregation; no pickle or executable artifacts."""

import hashlib
import json
import time

import numpy as np


def head_digest(checkpoint):
    return hashlib.sha256(json.dumps(checkpoint, sort_keys=True).encode()).hexdigest()


def infer_head(base, checkpoint, *, evidence_group_count, extra_features=()):
    if base.get("logits") is None:
        return base
    values = np.array(
        base["logits"]
        + [
            np.log1p(base.get("token_count") or base.get("model_input_tokens") or 0),
            np.log1p(evidence_group_count),
        ]
        + list(extra_features),
        dtype=float,
    )
    values = (values - np.array(checkpoint["scaler_mean"])) / np.array(checkpoint["scaler_scale"])
    if isinstance(checkpoint["weights"][0][0], list):
        for i, (weights, bias) in enumerate(zip(checkpoint["weights"], checkpoint["biases"])):
            values = values @ np.array(weights) + np.array(bias)
            if i < len(checkpoint["weights"]) - 1:
                values = np.maximum(values, 0)
    else:
        values = values @ np.array(checkpoint["weights"]).T + np.array(checkpoint["biases"])
    probabilities = np.exp(values - values.max())
    probabilities /= probabilities.sum()
    index = int(probabilities.argmax())
    return {
        **base,
        "encoder_logits": base["logits"],
        "logits": values.tolist(),
        "probabilities": probabilities.tolist(),
        "candidate": {0: "NO", 1: "YES", 2: "MAYBE"}[checkpoint["classes"][index]],
        "raw_score": float(probabilities[index]),
    }


class HeadNLI:
    def __init__(self, model, checkpoint, name):
        self.model, self.checkpoint = model, checkpoint
        self.config = {
            **model.config,
            "aggregation_head": name,
            "head_digest": head_digest(checkpoint),
        }

    def predict(self, question, evidence):
        start = time.perf_counter()
        base = self.model.predict(question, evidence)
        extra = ()
        if self.checkpoint.get("feature_schema"):
            from .features import SCHEMA, context_features

            if self.checkpoint["feature_schema"] != SCHEMA:
                raise ValueError("Unsupported feature schema")
            extra = context_features(question, evidence)
        result = infer_head(
            base,
            self.checkpoint,
            evidence_group_count=len({e.get("page", e.get("url")) for e in evidence}),
            extra_features=extra,
        )
        result["model_seconds"] = time.perf_counter() - start
        return result
