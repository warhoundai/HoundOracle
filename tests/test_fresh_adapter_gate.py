import importlib.util
import json
from pathlib import Path

import pytest


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts/assess_fresh_adapter.py"
    spec = importlib.util.spec_from_file_location("fresh_adapter_gate_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(tmp_path, monkeypatch, *, qualified=False, frozen_hash="a"):
    module = load_script()
    monkeypatch.chdir(tmp_path)
    previous = Path("research/experiments/campaign-38")
    previous.mkdir(parents=True)
    (previous / "screen.json").write_text(
        json.dumps({"screen_passed": True, "training": {"checkpoint_sha256": "a"}})
    )
    public = Path("research/experiments/campaign-39")
    public.mkdir(parents=True)
    (public / "frozen-policy.json").write_text(json.dumps({"adapter_sha256": frozen_hash}))
    (public / "risk-assessment.json").write_text(json.dumps({"qualified80": qualified}))
    monkeypatch.setattr(module, "PUBLIC", public)
    monkeypatch.setattr(
        module, "checked", lambda stage: pytest.fail("Locked labels must remain unread")
    )
    monkeypatch.setattr("sys.argv", ["assess_fresh_adapter.py", "locked"])
    return module


def test_failed_risk_gate_prevents_even_reading_locked_labels(tmp_path, monkeypatch):
    module = prepare(tmp_path, monkeypatch)
    with pytest.raises(PermissionError):
        module.main()


def test_changed_frozen_candidate_prevents_reading_locked_labels(tmp_path, monkeypatch):
    module = prepare(tmp_path, monkeypatch, qualified=True, frozen_hash="b")
    with pytest.raises(ValueError):
        module.main()
