import time

import pytest

from houndoracle import DocumentInput, Oracle
from houndoracle.calibration import Calibration, CalibrationBin


@pytest.fixture
def oracle():
    o = Oracle()
    # Synthetic control-flow fixture, NOT evidence of real-world calibration.
    o.calibration = Calibration(
        policy_id=o.policy_id,
        population="synthetic-unit-tests-only",
        supported_kinds=["release_year", "quantity", "not", "and", "or"],
        expires_at=time.time() + 3600,
        dataset_digest="0" * 64,
        bins=[CalibrationBin(lower=0, upper=1, correct=100, count=100)],
    )
    yield o
    o.store.close()


@pytest.fixture
def ingest(oracle):
    def add(text="Widget was released in 2024.", **kwargs):
        now = time.time()
        return oracle.store.ingest(
            DocumentInput.model_validate(
                {
                    "url": "https://example.org/releases",
                    "text": text,
                    "origin": "official",
                    "observed_at": now,
                    "expires_at": now + 3600,
                    "primary": True,
                    "reviewed": True,
                    **kwargs,
                }
            )
        )

    return add
