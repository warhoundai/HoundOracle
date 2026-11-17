import numpy as np

from houndoracle.experiments.features import context_features
from houndoracle.experiments.head import infer_head


def test_context_features_are_order_invariant_and_finite():
    evidence = [{"text": "The 2024 release."}, {"text": "Not released in 2023."}]
    a = context_features("Released in 2024?", evidence)
    assert a == context_features("Released in 2024?", list(reversed(evidence)))
    assert all(np.isfinite(context_features("", [])))
    assert a != context_features("Not released in 2023?", evidence)


def test_context_head_json_matches_matrix_inference():
    checkpoint = {
        "weights": [[1.0] * 6, [0.0] * 6, [-1.0] * 6],
        "biases": [0.0, 0.0, 0.0],
        "classes": [0, 1, 2],
        "scaler_mean": [0.0] * 6,
        "scaler_scale": [1.0] * 6,
    }
    result = infer_head(
        {"logits": [1.0, 2.0, 3.0], "token_count": 1},
        checkpoint,
        evidence_group_count=1,
        extra_features=[0.5],
    )
    assert result["candidate"] == "NO"
    assert np.isclose(sum(result["probabilities"]), 1)


def test_unreviewed_model_revision_is_rejected(tmp_path):
    import json

    import pytest

    from houndoracle.experiments.nli import OnnxNLI

    (tmp_path / "manifest.json").write_text(
        json.dumps({"model_id": "unreviewed/model", "revision": "unknown"})
    )
    with pytest.raises(ValueError, match="reviewed"):
        OnnxNLI(tmp_path)


def test_token_override_cannot_exceed_reviewed_model_limit(tmp_path):
    import json

    import pytest

    from houndoracle.experiments.nli import OnnxNLI

    manifest = {"model_id": "fixture", "revision": "test", "max_length": 256}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for invalid in [512, 31, True, 256.5]:
        with pytest.raises(ValueError, match="token envelope"):
            OnnxNLI(tmp_path, reviewed_manifest=manifest, max_length=invalid)
