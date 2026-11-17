# Contributing

Use Python 3.11+ and uv. Run `uv sync --extra dev --extra api --extra mcp --extra s3`,
`uv run pytest`, `uv run ruff check src tests examples benchmarks scripts`, and
`uv run python -m build`. Do not commit source credentials, private evidence,
uncleared model weights/datasets or downloaded research implementations.

Changes to question semantics, evidence eligibility, grouping, scores or stopping
must update the policy identity and invalidate prior calibration. Preserve the
uncalibrated abstaining baseline. Experimental improvements need locked-test,
matched-risk/coverage evidence before production adoption. Report negative results.
