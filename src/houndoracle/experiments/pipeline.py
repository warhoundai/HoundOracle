"""Serializable research calibration and conservative acceptance on a declared frozen population."""

import hashlib
import json
import math
import time

from .fit_calibration import predict


def apply(row, artifact):
    raw = row.get("raw_score")
    probability = (
        predict(artifact["mapping"], raw)
        if raw is not None and row.get("candidate") in ("YES", "NO")
        else None
    )
    eligible = (
        row.get("candidate") in ("YES", "NO")
        and not row.get("truncated", False)
        and probability is not None
        and probability >= 0.8
    )
    certificate = artifact["certificate"]
    upper = certificate.get("row_risk_upper", certificate.get("row_risk_upper_bootstrap"))
    valid_upper = (
        isinstance(upper, (int, float))
        and not isinstance(upper, bool)
        and math.isfinite(upper)
        and 0 <= upper <= 1
    )
    confidence = min(probability, 1 - upper) if eligible and valid_upper else None
    qualified = certificate.get("qualified") is True and valid_upper and upper <= 0.2
    accepted = eligible and qualified and confidence is not None and confidence >= 0.8
    return {
        **row,
        "decision": row["candidate"] if accepted else "MAYBE",
        "calibrated_correctness": probability,
        "calibration_qualified": qualified,
        "reason": "CALIBRATED_RESEARCH_ACCEPTANCE" if accepted else "RESEARCH_ABSTENTION",
        "confidence": confidence if accepted else None,
    }


class ResearchPipeline:
    """Explicit frozen-evidence interface. No central upload or production replacement."""

    def __init__(self, model, artifact):
        if model.config != artifact["model_config"]:
            raise ValueError("Calibration belongs to a different model policy")
        self.model, self.artifact = model, artifact
        self.cache = {}

    def verify_evidence(self, question, evidence, *, population, use_cache=True):
        if population != self.artifact["population"]:
            raise ValueError("Unsupported calibration population")
        start = time.perf_counter()
        key = hashlib.sha256(
            json.dumps(
                [question, evidence, self.model.config, self.artifact], sort_keys=True
            ).encode()
        ).hexdigest()
        if use_cache and key in self.cache:
            return {
                **self.cache[key],
                "cache_hit": True,
                "latency_seconds": time.perf_counter() - start,
            }
        raw = self.model.predict(question, evidence)
        result = apply(raw, self.artifact)
        result.update(cache_hit=False, latency_seconds=time.perf_counter() - start)
        # Feature/result reuse is exact and scoped to the frozen corpus; not evidence of itself.
        if use_cache:
            self.cache[key] = result
        return result
