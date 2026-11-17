# Local verification record

Verified 2026-11-15 in this workspace:

- Python 3.12.14: **70 tests passed**.
- Python 3.11.18 in a separate environment: **70 tests passed**.
- Ruff lint and formatting checks passed.
- Source distribution and wheel built successfully.
- Built wheel installed in a clean Python 3.11 environment with core dependencies
  only; local verification smoke passed without API/MCP/S3/model packages.
- Installed third-party dependency audit: no known vulnerabilities found after
  upgrading optional cryptography to 50.0.2. The unpublished Oracle package itself
  is not covered by the external vulnerability database.
- Synthetic benchmark: 160 test cases; default uncalibrated definitive coverage 0;
  synthetic-only calibrated coverage .75 with zero observed errors on 120
  definitive template-generated cases; 120 warm-cache hits. These are engineering
  smoke results, not independent real-world validation or proof of superiority.

Raw command outcomes are in `validation.json`; benchmark parameters, hashes,
latencies, intervals and caveats are in `../benchmarks/local-result.json`.
`dependency-licenses.json` records installed package declarations, including dev
packages. The remaining Starlette/httpx deprecation warning did not fail tests;
no warning is treated as evidence of factual reliability.

No live search, cloud provisioning, production model download, public repository
creation, GitHub release or OPS modification occurred. Public release gates remain
closed as described in `status.md` and `evaluation.md`.

Reproduce primary checks with `uv run python scripts/validate.py`. Reproduce the
second interpreter suite with:

```sh
UV_PROJECT_ENVIRONMENT=/tmp/houndoracle-py311 uv run --python 3.11 --locked \
  --extra dev --extra api --extra mcp --extra s3 pytest -q
```
