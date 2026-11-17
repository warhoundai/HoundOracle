"""Conservative two-model semantic cross-check, calibrated against both observed regimes."""

import time

from .fit_calibration import predict

POLICY = "fine-plus-base-semantic-guard-v1"


class GuardedCompiler:
    def __init__(self, fine, base):
        self.fine, self.base = fine, base
        self.config = {
            "policy": POLICY,
            "fine": fine.config,
            "semantic_guard": base.config,
            "minimum_guard_score": 0.5,
        }

    def predict(self, question, evidence):
        start = time.perf_counter()
        fine = self.fine.predict(question, evidence)
        guard = None
        if fine.get("candidate") in ["YES", "NO"] and not fine.get("truncated"):
            guard = self.base.predict(question, evidence)
        eligible = (
            guard is not None
            and guard.get("candidate") == fine.get("candidate")
            and (guard.get("raw_score") or 0) >= 0.5
            and not guard.get("truncated")
        )
        return {
            **fine,
            "semantic_guard_passed": bool(eligible),
            "semantic_guard_candidate": guard.get("candidate") if guard else None,
            "semantic_guard_raw_score": guard.get("raw_score") if guard else None,
            "model_seconds": time.perf_counter() - start,
        }


def proposed(row, artifact):
    eligible = (
        row.get("semantic_guard_passed")
        and row.get("candidate") in ["YES", "NO"]
        and not row.get("truncated")
    )
    probability = (
        min(predict(m, row["raw_score"]) for m in artifact["regime_mappings"].values())
        if eligible
        else None
    )
    return {
        "decision": row["candidate"]
        if probability is not None and probability >= artifact["minimum_confidence"]
        else "MAYBE",
        "estimated_correctness": probability,
    }


class GuardedVerifier:
    def __init__(self, model, artifact):
        if model.config != artifact["model_config"]:
            raise ValueError("Guard calibration belongs to a different model policy")
        self.model, self.artifact = model, artifact

    def verify_evidence(self, question, evidence, *, population, use_cache=False):
        if population != self.artifact["population"]:
            raise ValueError("Unsupported guarded-calibration population")
        start = time.perf_counter()
        row = self.model.predict(question, evidence)
        decision = proposed(row, self.artifact)
        qualified = self.artifact["certificate"]["qualified"] is True
        accepted = qualified and decision["decision"] in ["YES", "NO"]
        return {
            **row,
            "decision": decision["decision"] if accepted else "MAYBE",
            "confidence": decision["estimated_correctness"] if accepted else None,
            "calibrated_correctness": decision["estimated_correctness"],
            "calibration_qualified": qualified,
            "cache_hit": False,
            "latency_seconds": time.perf_counter() - start,
            "reason": "REGIME_GUARDED_ACCEPTANCE" if accepted else "GUARDED_ABSTENTION",
        }
