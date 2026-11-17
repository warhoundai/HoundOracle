"""Research evidence compiler: keep concatenated context and add deduplicated passage signals."""

import time

import numpy as np

from .head import head_digest, infer_head

POLICY = "concat-plus-passage-pool-v1"


def pooled_features(model, question, evidence, base, *, max_passages=8):
    unique = list(dict.fromkeys(e["text"] for e in evidence if e["text"]))
    truncated = len(unique) > max_passages or base.get("truncated", False)
    if not unique or base.get("probabilities") is None:
        return [0.0] * 9, truncated, 0
    values = []
    calls = 0
    for text in unique[:max_passages]:
        if len(evidence) == 1:
            result = base
        else:
            result = model.predict(question, [{"text": text}])
            calls += 1
        values.append(result["probabilities"])
        truncated = truncated or result.get("truncated", False)
    matrix = np.array(values, dtype=float)
    return (
        np.concatenate([matrix.max(axis=0), matrix.mean(axis=0), matrix.min(axis=0)]).tolist(),
        truncated,
        calls,
    )


class PairwiseHead:
    def __init__(self, model, checkpoint, *, max_passages=8):
        if (
            checkpoint.get("evidence_policy") != POLICY
            or checkpoint.get("max_passages") != max_passages
        ):
            raise ValueError("Head belongs to a different evidence compiler")
        self.model, self.checkpoint = model, checkpoint
        self.max_passages = max_passages
        self.config = {
            **model.config,
            "aggregation_head": "passage-pool-logistic",
            "evidence_policy": POLICY,
            "max_passages": max_passages,
            "head_digest": head_digest(checkpoint),
        }

    def predict(self, question, evidence):
        start = time.perf_counter()
        base = self.model.predict(question, evidence)
        extra, truncated, calls = pooled_features(
            self.model, question, evidence, base, max_passages=self.max_passages
        )
        result = infer_head(
            base,
            self.checkpoint,
            evidence_group_count=len({e.get("page", e.get("url")) for e in evidence}),
            extra_features=extra,
        )
        return {
            **result,
            "truncated": truncated,
            "model_seconds": time.perf_counter() - start,
            "additional_passage_inferences": calls,
        }
