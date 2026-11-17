"""Safe serialized three-class score adjustment; correctness requires separate qualification."""

import numpy as np

SCHEMA = "three-relation-logits-NO-YES-MAYBE-v1"


def apply(row, artifact):
    if artifact.get("feature_schema") != SCHEMA or artifact.get("classes") != [0, 1, 2]:
        raise ValueError("Unsupported relation-score schema or label ordering")
    z = np.asarray(row["logits"], dtype=float)
    mean = np.asarray(artifact["mean"])
    scale = np.asarray(artifact["scale"])
    weights = np.asarray(artifact["weights"])
    bias = np.asarray(artifact["bias"])
    if (
        z.shape != (3,)
        or mean.shape != (3,)
        or scale.shape != (3,)
        or weights.shape != (3, 3)
        or bias.shape != (3,)
        or not all(np.isfinite(v).all() for v in [z, mean, scale, weights, bias])
        or (scale <= 0).any()
    ):
        raise ValueError("Malformed or nonfinite class calibration state")
    logits = weights @ ((z - mean) / scale) + bias
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return row | {
        "candidate": ["NO", "YES", "MAYBE"][int(probabilities.argmax())],
        "raw_score": float(probabilities.max()),
        "logits": logits.tolist(),
        "relation_scores": probabilities.tolist(),
        "calibrated_correctness": None,
        "production_qualified": False,
    }
