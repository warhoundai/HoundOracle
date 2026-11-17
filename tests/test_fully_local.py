import json
import socket

import pytest

from houndoracle import Oracle
from houndoracle.local_index import import_documents


def test_required_local_path_ignores_hosted_api_environment_and_cannot_send_queries(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("EXA_API_KEY", "must-never-be-used")

    def denied(*a, **k):
        raise AssertionError("Offline inference attempted networking")

    monkeypatch.setattr(socket, "create_connection", denied)
    oracle = Oracle()
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps(
            {
                "url": "file:///local/release.txt",
                "text": "Widget was released in 2024.",
                "reviewed": True,
                "primary": True,
                "public": True,
            }
        )
        + "\n"
    )
    imported = import_documents(oracle.store, path)
    assert imported["hosted_api_calls"] == 0
    assert oracle.search("Widget")
    assert (
        oracle.verify(
            "Was Widget released in 2024?", allow_network=True, max_cost_usd=1
        ).decision.value
        == "MAYBE"
    )
    assert oracle.web_search("Widget")["source"] == "local"
    with pytest.raises(PermissionError, match="LOCAL_ONLY"):
        oracle.web_search("Unknown", allow_network=True, max_cost_usd=1)
    assert oracle.store.documents()[0]["data"]
    with pytest.raises(ValueError, match="external search provider"):
        Oracle(provider=object())
    oracle.store.close()


def test_search_common_words_do_not_suppress_discovery_with_irrelevant_hits(tmp_path):
    oracle = Oracle()
    p = tmp_path / "index.jsonl"
    p.write_text(
        json.dumps({"url": "file:///widget", "text": "Widget was released in 2024."}) + "\n"
    )
    import_documents(oracle.store, p)
    assert oracle.search("Was Widget released in 2024?")
    assert oracle.search("Was UnfindableZXQ90210 mentioned in records?") == []
    oracle.store.close()


def test_cli_has_no_hosted_search_escape_hatch(monkeypatch):
    from houndoracle.cli import main

    monkeypatch.setattr("sys.argv", ["houndoracle", "search", "private question", "--online"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
