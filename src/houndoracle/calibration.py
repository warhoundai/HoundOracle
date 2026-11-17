"""Population-specific selective calibration. Not a per-answer guarantee."""

import hashlib
import json
import math
import time

from pydantic import Field

from .models import Strict


class CalibrationBin(Strict):
    lower: float = Field(ge=0, le=1)
    upper: float = Field(gt=0, le=1)
    correct: int = Field(ge=0)
    count: int = Field(gt=0)

    def estimate(self):
        if self.correct > self.count or self.upper <= self.lower:
            raise ValueError("Invalid calibration bin")
        # One-sided 95% Wilson lower bound, reported as an empirical gate.
        p, n, z = self.correct / self.count, self.count, 1.6448536269514722
        return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (
            1 + z * z / n
        )


class Calibration(Strict):
    schema_version: int = 1
    policy_id: str
    population: str = Field(min_length=1)
    supported_kinds: list[str]
    expires_at: float
    dataset_digest: str = Field(min_length=64, max_length=64)
    bins: list[CalibrationBin]

    @property
    def version(self):
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    def confidence(self, score, kinds, policy_id):
        if (
            self.schema_version != 1
            or self.policy_id != policy_id
            or self.expires_at <= time.time()
        ):
            return None
        if not kinds.issubset(set(self.supported_kinds)):
            return None
        ordered = sorted(self.bins, key=lambda b: b.lower)
        for i, b in enumerate(ordered):
            b.estimate()
            if i and ordered[i - 1].upper > b.lower:
                return None
        matches = [
            b for b in ordered if b.lower <= score < b.upper or (score == 1 and b.upper == 1)
        ]
        return matches[0].estimate() if len(matches) == 1 else None

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.model_validate(json.load(f))


class GeneralCalibration(Strict):
    """A full-pipeline map; research artifacts cannot authorize serving decisions."""

    schema_version: str = "general-1"
    policy_id: str
    population: str = Field(min_length=1)
    supported_kinds: list[str]
    expires_at: float
    dataset_digest: str = Field(min_length=64, max_length=64)
    method: str = "logistic"
    weight: float
    bias: float
    threshold: float = Field(0.8, ge=0.8, le=1)
    risk_lower_bound: float = Field(ge=0, le=1)
    risk_groups: int = Field(ge=30)
    qualification_status: str = "research"
    audit_references: list[str] = Field(default_factory=list)

    @property
    def version(self):
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    def estimate(self, score):
        if self.schema_version != "general-1" or self.method != "logistic":
            return None
        if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1:
            return None
        bounded = max(1e-7, min(1 - 1e-7, score))
        value = self.weight * math.log(bounded / (1 - bounded)) + self.bias
        return 1 / (1 + math.exp(-max(-50, min(50, value))))

    def confidence(self, score, kinds, policy_id):
        if (
            self.qualification_status != "audited"
            or not self.audit_references
            or self.expires_at <= time.time()
            or self.policy_id != policy_id
            or not kinds.issubset(set(self.supported_kinds))
            or self.risk_lower_bound < self.threshold
        ):
            return None
        estimate = self.estimate(score)
        if estimate is None or estimate < self.threshold:
            return None
        return min(estimate, self.risk_lower_bound)


def load_calibration(path):
    """Read legacy calibration or an explicitly scoped general pipeline artifact."""
    with open(path) as handle:
        data = json.load(handle)
    if data.get("schema_version") == "general-1":
        return GeneralCalibration.model_validate(data)
    return Calibration.model_validate(data)
