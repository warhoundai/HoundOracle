import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "bundle", Path(__file__).parents[1] / "scripts/cloud_bundle.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_deferred_citation_dependency_is_rejected_before_paid_inference():
    files = {
        "houndoracle/experiments/vertex.py": "def definite():\n from .citations import citations_match\n return citations_match\n",
        "runner.py": "from houndoracle.experiments.vertex import definite\n",
    }
    with pytest.raises(ValueError, match="Missing bundled dependency.*citations"):
        module.validate_bundle(files)
    files["houndoracle/experiments/citations.py"] = "def citations_match(): return True\n"
    module.validate_bundle(files)
