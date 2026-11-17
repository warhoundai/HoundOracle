"""Scoped local insufficiency selection; never turns an evidence absence into NO."""

import hashlib
import json

from .fit_calibration import predict


def policy_digest(artifact):
    return hashlib.sha256(
        json.dumps(artifact, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def would_abstain(row, artifact):
    return (
        row.get("candidate") == "MAYBE"
        and not row.get("truncated", False)
        and row.get("raw_score") is not None
        and predict(artifact["mapping"], row["raw_score"]) >= artifact["threshold"]
    )


class AbstentionGate:
    def __init__(self, artifact):
        self.artifact = artifact

    def allows(self, row, *, model_config, verification_artifact):
        if (
            model_config != self.artifact["model_config"]
            or policy_digest(verification_artifact) != self.artifact["verification_policy_digest"]
        ):
            return False
        return self.artifact["risk_certificate"]["qualified"] is True and would_abstain(
            row, self.artifact
        )
