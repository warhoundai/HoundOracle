import pytest

from houndoracle.experiments.paired_risk import paired_improvement_lower


def test_zero_discordance_retains_unobserved_error_uncertainty():
    bound = paired_improvement_lower([0] * 600)
    assert -0.007 < bound < -0.006
    assert paired_improvement_lower([0] * 30) < -0.10
    assert paired_improvement_lower([1] * 600) > 0.98
    assert paired_improvement_lower([-1] * 600) == -1
    with pytest.raises(ValueError):
        paired_improvement_lower([])
