"""Immutable historical source references; neither a truth nor content-match proof."""

from urllib.parse import urlencode


def wikipedia_revision_reference(page, revision_id):
    if not isinstance(page, str) or not 1 <= len(page) <= 255 or any(ord(c) < 32 for c in page):
        raise ValueError("A bounded single-line Wikipedia page title is required")
    if isinstance(revision_id, bool) or not isinstance(revision_id, (str, int)):
        raise TypeError("A positive numeric revision identifier is required")
    revision = str(revision_id)
    if (
        not revision.isascii()
        or not revision.isdigit()
        or not 1 <= len(revision) <= 20
        or int(revision) == 0
    ):
        raise ValueError("A positive numeric revision identifier is required")
    return "https://en.wikipedia.org/w/index.php?" + urlencode({"title": page, "oldid": revision})
