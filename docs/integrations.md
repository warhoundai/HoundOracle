# Integration examples

## Offline Python

Run `uv run python examples/offline.py`. Use `DocumentInput` through the local SDK
or curator CLI to import actual evidence. Controlled-language record formats are
`Widget was released in 2024.` and `Widget has mass 1000 g.` as complete lines.
Do not rewrite arbitrary prose into these formats without retaining and reviewing
the original source/extraction trail. The current compiler does not qualify that
rewriting process.

Use typed quantities and Boolean relationships via `verify_claim(Claim(...))`.
An arbitrary supplied claim cannot override the meaning of a natural question.
Store imports preserve dependency edges but reset review, authority and sharing.

## Self-hosted online with HoundSearch

```python
import os
from houndoracle import Oracle, Store
from houndoracle.network import HoundSearchAdapter

provider = HoundSearchAdapter(
    os.environ['HOUNDSEARCH_ENDPOINT'],
    os.environ['HOUNDSEARCH_API_KEY'],
    request_cost_upper_usd=float(os.environ['HOUNDSEARCH_REQUEST_COST_CAP_USD']),
)
oracle = Oracle(Store('oracle.sqlite'), provider=provider)
result = oracle.verify(
    'Was Widget released in 2024?', allow_network=True,
    max_requests=2, max_seconds=10,
    max_cost_usd=float(os.environ['HOUNDORACLE_QUERY_COST_CAP_USD']),
)
```

The positive adapter cost cap is explicit, must include downstream calls, and is
sent to HoundSearch's own budget controller. No amount is inferred or prepaid by
Oracle. An explicit private endpoint allowlist is available for a local deployment;
remote endpoints require HTTPS. Live hits are unreviewed evidence, not automatic
permission to return YES/NO. The contract has been tested with a mock response;
a running HoundSearch deployment has not been qualified here.

HoundSearch already integrates the HoundCrawler extraction stack. Oracle does not
import or alter its Python 3.12 package. HoundBrowser output can be imported as
source evidence through the same documented document schema. Native direct adapters
for every released project are not claimed.

## REST and MCP

REST instance construction: `create_app(oracle, api_key=...)` from
`houndoracle.api`. Configure an Oracle with a provider in the embedding application
to enable online requests. CLI `serve` uses the offline engine by default.

POST `/v1/verify`: `{"question":"Was Widget released in 2024?"}` with bearer key.
POST `/v1/search`: `{"query":"Widget","limit":10}` returns discovery results.
POST `/v1/jobs` accepts verification input, returns an ID; GET status and POST
`/{pause,resume,cancel}` control cooperative execution with cumulative reservations.
Reconcile interrupted processes before replaying potentially billed requests.

MCP stdio command: `uv run houndoracle --database oracle.sqlite mcp`. Tools are
`search`, `verify`, `explain`. MCP deliberately exposes local evidence only in this
preview; its host process permissions authorize database access.

## Optional S3 snapshots

Install `[s3]`, provide a boto3 S3 client and an existing private/versioned bucket,
and use `snapshots.prepare` with a reviewed license allowlist. Generate/protect an
Ed25519 signing key outside source control. Call `S3Snapshots.publish(...,
authorized=True, allowed_licenses=...)` only for vetted public evidence. Consumers
pin the trusted public key and manifest digest, then explicitly authorize download.
Remote review remains untrusted unless the operator enables `trust_remote_review`.

This is a library integration, not a deployed central API. Signed snapshots prove
integrity and curator origin, not truth or legal permission. Revocation-feed and
central quarantine/curator service qualification are required before cloud release.
