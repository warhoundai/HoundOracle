import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from houndoracle.experiments.journal import JournalError, TaskJournal, TimeBudgetError


def journal(path, worker="one", cap=2, dollars=0.2):
    return TaskJournal(
        path,
        configuration={"dataset": "v1", "policy": "frozen"},
        maximum_calls=cap,
        authorized_usd=dollars,
        worker_id=worker,
    )


def test_crash_recovery_keeps_reservation_and_never_retries(tmp_path):
    path = tmp_path / "journal.sqlite"
    first = journal(path)
    first.add("task", {"question": "X", "evidence": [1]})
    assert first.claim("task")
    first.reserve("task", 0.1, remaining_seconds=70, required_seconds=65)
    first.close()
    second = journal(path, worker="two")
    with pytest.raises(PermissionError):
        second.recover_worker("one")
    assert second.recover_worker("one", confirmed_stopped=True) == 1
    assert not second.claim("task")
    assert second.result("task") is None
    assert second.summary()["token_cost_bounds_usd"] == [0, 0.1]
    assert second.summary()["unknown_usage_calls"] == 1
    assert os.stat(path).st_mode & 0o077 == 0
    second.close()


def test_completed_results_and_input_policy_integrity(tmp_path):
    path = tmp_path / "complete.sqlite"
    j = journal(path)
    j.add("task", {"claim": "Not X"})
    assert j.claim("task")
    call = j.reserve("task", 0.1, remaining_seconds=100, required_seconds=65)
    with pytest.raises(JournalError):
        j.finish("task", {"decision": "YES"})
    j.settle(call, actual_cost=0.01)
    j.finish("task", {"decision": "NO"})
    assert not j.claim("task")
    assert j.result("task") == {"decision": "NO"}
    with pytest.raises(ValueError):
        j.add("task", {"claim": "X"})
    j.close()
    with pytest.raises(ValueError):
        journal(path, cap=3)
    resumed = journal(path)
    assert resumed.result("task") == {"decision": "NO"}
    resumed.close()


def test_time_preflight_defer_and_failed_usage(tmp_path):
    j = journal(tmp_path / "time.sqlite")
    j.add("task", {})
    j.claim("task")
    with pytest.raises(TimeBudgetError):
        j.reserve("task", 0.1, remaining_seconds=10, required_seconds=65)
    assert j.summary()["reserved_calls"] == 0
    j.defer("task")
    assert j.claim("task")
    call = j.reserve("task", 0.1, remaining_seconds=100, required_seconds=65)
    j.settle(call)
    j.finish("task", {"decision": "MAYBE"}, failure="PROVIDER_TIMEOUT")
    assert j.summary()["tasks"] == {"FAILED": 1}
    assert not j.claim("task")
    assert j.summary()["reserved_usd"] == 0.1
    j.close()


def test_atomic_caps_across_workers(tmp_path):
    path = tmp_path / "concurrent.sqlite"
    first = journal(path, worker="first", cap=3, dollars=0.3)
    second = journal(path, worker="second", cap=3, dollars=0.3)
    for i in range(10):
        first.add(str(i), {"id": i})

    def reserve(i):
        worker = first if i % 2 else second
        assert worker.claim(str(i))
        try:
            worker.reserve(str(i), 0.1, remaining_seconds=100, required_seconds=65)
            return 1
        except PermissionError:
            return 0

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(reserve, range(10))) == 3
    assert first.summary()["reserved_calls"] == 3
    first.close()
    second.close()


def test_cost_overrun_is_preserved_and_halts(tmp_path):
    j = journal(tmp_path / "overrun.sqlite")
    j.add("a", {})
    j.claim("a")
    call = j.reserve("a", 0.1, remaining_seconds=100, required_seconds=65)
    with pytest.raises(JournalError, match="exceeded"):
        j.settle(call, actual_cost=0.11)
    assert j.summary()["reported_token_cost_usd"] == 0.11
    j.add("b", {})
    with pytest.raises(JournalError, match="halted"):
        j.claim("b")
    j.close()


def test_abrupt_process_exit_preserves_uncertain_call(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "killed.sqlite"
    source = """import os,sys
from houndoracle.experiments.journal import TaskJournal
j=TaskJournal(sys.argv[1],configuration={'dataset':'v1','policy':'frozen'},maximum_calls=2,authorized_usd=.2,worker_id='killed')
j.add('request',{'question':'X'});j.claim('request')
j.reserve('request',.1,remaining_seconds=100,required_seconds=65)
os._exit(17)
"""
    child = subprocess.run([sys.executable, "-c", source, str(path)], check=False, timeout=10)
    assert child.returncode == 17
    resumed = journal(path, worker="resumer")
    assert resumed.recover_worker("killed", confirmed_stopped=True) == 1
    assert resumed.summary()["reserved_calls"] == 1
    assert not resumed.claim("request")
    resumed.close()
