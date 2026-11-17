"""The experimental specialist can abstain, never add definitive answers."""

from types import SimpleNamespace

import pytest
from test_router_optimizations import Gemini, Oracle

from houndoracle.experiments.abstention import AbstentionGate, policy_digest
from houndoracle.experiments.router import OracleFirstRouter


def setup(candidate="MAYBE", qualified=True):
    oracle, gemini = Oracle(), Gemini()
    oracle.verify_evidence = lambda *a, **k: {"decision": "MAYBE", "confidence": None}
    calls = []
    model = SimpleNamespace(config={"revision": "test"})
    model.predict = lambda *a: (
        calls.append(a) or {"candidate": candidate, "raw_score": 0.99, "truncated": False}
    )
    specialist = SimpleNamespace(model=model, artifact={"version": 2})
    gate = AbstentionGate(
        {
            "model_config": model.config.copy(),
            "verification_policy_digest": policy_digest(specialist.artifact),
            "mapping": {"method": "isotonic", "x": [0, 1], "y": [0, 1]},
            "threshold": 0.9,
            "risk_certificate": {"qualified": qualified},
        }
    )
    return oracle, gemini, specialist, gate, calls


@pytest.mark.parametrize(
    "complete,validated,bypass", [(False, True, False), (True, False, False), (True, True, True)]
)
def test_requires_closed_validated_scope(complete, validated, bypass):
    o, g, s, gate, calls = setup()
    router = OracleFirstRouter(
        o,
        g,
        complete_evidence=complete,
        bypass=bypass,
        insufficiency_specialist=s,
        insufficiency_gate=gate,
    )
    assert (
        router.answer("Claim", ["evidence"], proposition_validated=validated)["source"]
        == "gemini_fallback"
    )
    assert not calls and g.calls == 1


def test_preserves_definitive_path_and_never_uses_specialist_truth():
    o, g, s, gate, calls = setup("YES")
    router = OracleFirstRouter(
        o, g, complete_evidence=True, insufficiency_specialist=s, insufficiency_gate=gate
    )
    assert (
        router.answer("Claim", ["evidence"], proposition_validated=True)["source"]
        == "gemini_fallback"
    )
    o.verify_evidence = lambda *a, **k: {"decision": "NO", "confidence": 0.95}
    calls.clear()
    result = router.answer("Claim", ["evidence"], proposition_validated=True)
    assert result["decision"] == "NO" and not calls and g.calls == 1


def test_scope_binding_and_qualification_fail_closed():
    o, g, s, gate, calls = setup(qualified=False)
    router = OracleFirstRouter(
        o, g, complete_evidence=True, insufficiency_specialist=s, insufficiency_gate=gate
    )
    router.answer("Claim", ["evidence"], proposition_validated=True)
    assert not calls
    gate.artifact["risk_certificate"]["qualified"] = True
    s.artifact["version"] = 3
    router.answer("Claim", ["evidence"], proposition_validated=True)
    assert g.calls == 2


def test_qualified_specialist_only_returns_maybe():
    o, g, s, gate, calls = setup()
    router = OracleFirstRouter(
        o, g, complete_evidence=True, insufficiency_specialist=s, insufficiency_gate=gate
    )
    result = router.answer("Claim", ["evidence"], proposition_validated=True)
    assert result["source"] == "calibrated_insufficiency_specialist"
    assert result["decision"] == "MAYBE" and result["confidence"] is None
    assert g.calls == 0 and len(calls) == 1
