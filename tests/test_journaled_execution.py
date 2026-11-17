import time

from test_experiments import fake_client

from houndoracle.experiments.execution import JournaledGemini, execute_task
from houndoracle.experiments.journal import TaskJournal
from houndoracle.experiments.vertex import VertexConfig


def test_provider_uses_durable_reservation_and_skips_finished_work(tmp_path):
    journal = TaskJournal(
        tmp_path / "tasks.sqlite",
        configuration={"model": "fixture"},
        maximum_calls=2,
        authorized_usd=1,
    )
    client = fake_client()
    provider = JournaledGemini(
        VertexConfig(project="fixture"), journal, permission_scope="public-fixture", client=client
    )

    def operation():
        output = provider.answer("Claim", [])
        return {"decision": output["answer"]["decision"], "failure": None}

    one = execute_task(
        journal, provider, "one", {"question": "Claim"}, operation, deadline=time.monotonic() + 130
    )
    assert one["state"] == "SUCCEEDED" and client.models.calls == 2
    assert journal.summary()["reserved_calls"] == 1
    restored = execute_task(
        journal, provider, "one", {"question": "Claim"}, operation, deadline=time.monotonic() + 130
    )
    assert not restored["executed"] and client.models.calls == 2
    deferred = execute_task(
        journal, provider, "two", {"question": "Claim"}, operation, deadline=time.monotonic() + 10
    )
    assert deferred["state"] == "DEFERRED" and journal.summary()["reserved_calls"] == 1
    journal.close()


def test_provider_error_is_recorded_and_reservation_not_refunded(tmp_path):
    journal = TaskJournal(
        tmp_path / "error.sqlite", configuration={}, maximum_calls=1, authorized_usd=1
    )
    client = fake_client()

    def fail(**kwargs):
        raise TimeoutError("Fixture provider failure")

    client.models.generate_content = fail
    provider = JournaledGemini(
        VertexConfig(project="fixture"), journal, permission_scope="public-fixture", client=client
    )
    result = execute_task(
        journal,
        provider,
        "task",
        {},
        lambda: provider.answer("Claim", []),
        deadline=time.monotonic() + 130,
    )
    assert result["state"] == "FAILED" and result["result"]["decision"] == "MAYBE"
    assert journal.summary()["unknown_usage_calls"] == 1
    assert journal.summary()["token_cost_bounds_usd"][1] > 0
    assert not journal.claim("task")
    journal.close()


def test_provider_and_permission_changes_cannot_reuse_a_campaign(tmp_path):
    import pytest

    journal = TaskJournal(
        tmp_path / "bound.sqlite", configuration={}, maximum_calls=2, authorized_usd=1
    )
    JournaledGemini(
        VertexConfig(project="fixture"), journal, permission_scope="tenant-a", client=fake_client()
    )
    with pytest.raises(ValueError, match="permission scope"):
        JournaledGemini(
            VertexConfig(project="fixture"),
            journal,
            permission_scope="tenant-b",
            client=fake_client(),
        )
    with pytest.raises(ValueError, match="Provider settings"):
        JournaledGemini(
            VertexConfig(project="other-project"),
            journal,
            permission_scope="tenant-a",
            client=fake_client(),
        )
    journal.close()


def test_checkpoint_must_acknowledge_reservation_before_network(tmp_path):
    import pytest

    journal = TaskJournal(
        tmp_path / "checkpoint.sqlite", configuration={}, maximum_calls=1, authorized_usd=1
    )
    client = fake_client()

    class FailedCheckpoint:
        def commit(self, journal):
            raise RuntimeError("checkpoint unavailable")

    provider = JournaledGemini(
        VertexConfig(project="fixture"),
        journal,
        permission_scope="public-fixture",
        client=client,
        checkpoint=FailedCheckpoint(),
    )
    with pytest.raises(RuntimeError, match="checkpoint unavailable"):
        execute_task(
            journal,
            provider,
            "task",
            {},
            lambda: provider.answer("Claim", []),
            deadline=time.monotonic() + 180,
        )
    # countTokens is an unbilled SDK call; generation never starts without the durable ACK.
    assert client.models.calls <= 1
    assert journal.summary()["reserved_calls"] == 1
    assert journal.summary()["unknown_usage_calls"] == 1
    journal.close()
