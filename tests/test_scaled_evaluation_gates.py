import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "scaled_eval", Path(__file__).parents[1] / "scripts/evaluate_scaled_training.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_failed_screen_and_unqualified_model_never_consume_new_risk_or_test(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "PUBLIC", tmp_path)

    def forbidden(*a, **k):
        raise AssertionError("Must not run inference before acceptance gates")

    monkeypatch.setattr(module, "score", forbidden)
    (tmp_path / "screen.json").write_text(json.dumps({"screen_passed": False}))
    with pytest.raises(PermissionError, match="Failed development"):
        module.qualify()
    (tmp_path / "locked-authorization.json").write_text(json.dumps({"qualified": False}))
    with pytest.raises(PermissionError, match="No qualified"):
        module.locked()
    (tmp_path / "locked-authorization.json").write_text(
        json.dumps({"qualified": True, "file_sha256": {"screen.json": "not-the-frozen-hash"}})
    )
    with pytest.raises(ValueError, match="Frozen selection changed"):
        module.locked()
