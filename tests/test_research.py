import importlib.util
from pathlib import Path

from houndoracle.budget import Budget
from houndoracle.evaluators import LogisticAggregator


def test_logistic_fit():
    model = LogisticAggregator()
    model.fit([([1, 0, 0], 1), ([0, 1, 0], 0)] * 20)
    assert model.weights[0] > 1 and model.weights[1] < -1
    assert model.version != LogisticAggregator().version


def test_fire_style_zero_search_and_evidence_gate():
    spec = importlib.util.spec_from_file_location(
        "fire_style", Path(__file__).parents[1] / "benchmarks/fire_style.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def generate(**kwargs):
        return {"answer": "YES"}

    def search(query):
        raise AssertionError("Zero-search decision")

    result = module.run(
        "claim", generate, search, Budget(5, 1, 0), generation_cost_upper=0, search_cost_upper=0
    )
    assert result["decision"] == "YES"
    result = module.run(
        "claim",
        generate,
        search,
        Budget(5, 1, 0),
        generation_cost_upper=0,
        search_cost_upper=0,
        evidence_required=True,
    )
    assert result["decision"] == "MAYBE"
