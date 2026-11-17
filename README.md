# HoundOracle

Development preview of WarHound AI's local-first factual verification infrastructure
for AI agents. Apache-2.0 code; evidence and model licenses remain separate.

**No accuracy, cost or latency advantage has been established.** The default engine
returns MAYBE until a deployment-specific calibration artifact qualifies its decisions.
Current deterministic extraction supports controlled release-year and quantity records,
not unrestricted natural-language factual verification. An opt-in
[local general-proposition pipeline](docs/general-verification.md) is available
for research, with policy-bound qualification still pending.

Oracle runs locally with internet-enabled web discovery: see [Google and other web search](docs/web-search.md).
[WarHound tool integrations](docs/warhound-tools.md) connect your existing local services.
[Experimental local text-CAPTCHA recognition](docs/captcha.md) and a separate
[Google reCAPTCHA v2 helper](docs/google-recaptcha.md) are available; reliable Google completion is not established.
For disconnected use, see [offline self-hosting instructions](docs/self-hosting.md)
and [EC2 offline deployment results](research/experiments/local-22/README.md).

```sh
uv sync --extra dev --extra api --extra mcp
uv run houndoracle --database oracle.sqlite verify 'Was Widget released in 2024?'
uv run python examples/offline.py
uv run pytest
uv run python benchmarks/run.py --output benchmarks/local-result.json
```

```python
from houndoracle import Oracle, Store
oracle = Oracle(Store('oracle.sqlite'))
result = oracle.verify('Was Widget released in 2024?')
print(result.decision, result.reason, result.citations)
```

The Python SDK supports `search`, `verify`, `verify_claim`, and `explain`. Discovery
results do not imply verification. Missing evidence, unsupported semantics, expired
budgets, conflicts and operational failures produce MAYBE with separate reason codes.
INVALID is reserved for clearly non-propositional requests.

`ingest document.json` is an explicit local curator operation. Documents contain URL,
exact text, origin, observation/expiry times, review status, primary-source status,
license, sharing permission and optional upstream identifiers. A source hash proves
snapshot integrity, not truth. Imports lose trust and sharing permissions by default.
Never mark arbitrary search results reviewed merely to obtain a definitive answer.

```sh
export HOUNDORACLE_API_KEY='set-a-long-random-instance-secret'
uv run houndoracle --database oracle.sqlite serve  # 127.0.0.1:8083
uv run houndoracle --database oracle.sqlite mcp    # stdio, local process access
```

REST exposes `/v1/search`, `/v1/verify`, `/v1/explanations/{id}`, `/v1/jobs`, job
status and pause/resume/cancel. Bearer authentication is mandatory except `/health`.
Use a separate instance/store per tenant. Do not expose the SQLite file or MCP stdio
to untrusted processes. Private questions remain local unless an operator configures
a provider **and** a request explicitly allows network access.

HoundSearch integrates through an optional HTTP adapter; its Python 3.12 runtime and
released projects are unchanged. HoundCrawler, HoundScraper and HoundBrowser can
supply documents through that existing retrieval stack; native Oracle-specific
adapters are not yet qualified. See [architecture](docs/architecture.md),
[research](docs/research.md), [evaluation](docs/evaluation.md),
[licensing](docs/rights.md), [integrations](docs/integrations.md),
[verification](docs/verification.md) and [status](docs/status.md).

The follow-up [optimization report](research/engineering-report.md) contains an
independent-label historical benchmark, trained aggregation checkpoints, scoped
calibration and negative results. Gemini execution was quota-blocked; no
Gemini-relative performance advantage or production release is claimed.

The optional Google reCAPTCHA helper and its live test harness are separately AGPL-3.0-only; see [their license and setup](integrations/recaptcha-local/README.md). They are excluded from the Apache-licensed Python wheel.

The [project page source](site/index.html) and [publication status](docs/publication.md)
are prepared for an honest public research preview.
