from urllib.parse import parse_qs, urlsplit

import pytest

from houndoracle.experiments.snapshot import wikipedia_revision_reference


def test_historical_reference_keeps_revision_and_title_as_data():
    url = wikipedia_revision_reference("A & B?redirect=elsewhere", 139926649)
    parts = urlsplit(url)
    assert parts.hostname == "en.wikipedia.org" and parts.path == "/w/index.php"
    assert parse_qs(parts.query) == {"title": ["A & B?redirect=elsewhere"], "oldid": ["139926649"]}


@pytest.mark.parametrize("revision", [True, 0, -1, "latest", "12&redirect=evil", "١٢", None])
def test_invalid_revision_cannot_silently_fall_back_to_current_page(revision):
    with pytest.raises((ValueError, TypeError)):
        wikipedia_revision_reference("Title", revision)
