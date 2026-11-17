# Local primary-record tools

The experimental `ReleaseRecords` tool retrieves the official Node.js release
index and evaluates exact source-scoped propositions using calendar arithmetic,
version identity and three-valued Boolean logic. It runs on the user's machine,
requires no cloud account, paid search API, generative model or private-question
upload. Internet retrieval downloads a fixed public index; question text is never
sent to that website. Cached snapshots work offline until their TTL expires.

**These are record proofs, not calibrated general-world verification verdicts.**
`confidence` is null and `production_qualified` is false. The existing general
`Oracle.verify` confidence/provenance gates remain in force. Callers must retain
the returned scope and reason, not silently promote this output to an arbitrary
factual claim. The index's date and LTS fields must not be confused with an
independently established original announcement date or historical LTS status.

```python
from houndoracle.tools.releases import ReleaseRecords

records = ReleaseRecords("release-evidence.sqlite")
try:
    result = records.verify(
        "Was Node.js v22.0.0 released before Node.js v23.0.0?",
        allow_network=True,
    )
    print(result["decision"], result["scope"], result["citations"])
finally:
    records.close()
```

Other supported forms include exact dates, release-index years, day differences,
security-fix flags, current index LTS markings, major-version comparisons,
explicit negation and unmixed conjunctions/disjunctions. Unsupported questions,
missing records/fields, stale snapshots, failed retrieval and conflicting record
schemas return MAYBE or fail source ingestion; absence is not proof of falsehood.
Prerelease tags preserve exact identity and are not aliases for stable versions.

```sh
houndoracle --database local.sqlite release-record \
  'Was Node.js v22.0.0 released on 2024-04-24?' --allow-network
```

Direct retrieval uses the existing DNS-pinned HTTPS boundary. Environments that
require an operator-configured proxy may explicitly add `--use-environment-proxy`.
That transport accepts only fixed official index/announcement paths, blocks
redirects, checks public DNS and caps downloads at 2 MB/10 seconds. The configured
proxy is a trusted network dependency; it is not a self-hosted Google index.

For agent interfaces, pass the same `records` instance as `release_records=` to
`houndoracle.api.create_app(...)` or `houndoracle.mcp.server(...)`. REST exposes
`POST /v1/tools/release-records` with `question` and `allow_network`; MCP registers
`release_record_verify`. REST retains instance-key authentication. Offline Oracle
instances reject network permission even on this tool. The operator owns/ closes
the records instance separately from the main evidence store.

SQLite stores immutable content-addressed snapshot bodies and a current pointer,
not previously generated answers. Each query recompiles exact logic, checks
freshness and integrity, and returns exact source spans and calculation operands.
New snapshots replace the active pointer; old evidence is retained for audit.
Content hashes establish snapshot integrity, not the publisher's factual truth.
Source excerpts and snapshots are not redistributed with benchmark metadata;
upstream data licensing and independent provenance review remain release gates.
