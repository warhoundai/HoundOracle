# Primary-record tool: measured pilot

The local tool matched all 112 mechanically derived reference answers. This exceeds
95% **for this source-scoped record-relation pilot only**. The original general
fact-verification benchmark has not improved to 95%; this new test must not replace
or be pooled with it. No production/general-confidence qualification is granted.

| Local SDK mode | Correct verdicts | Definitive coverage | Correct definitive | P50 ms | P95 ms | Source requests |
|---|---:|---:|---:|---:|---:|---:|
| cold | 112/112 | 96/112 | 96 | 95.07 | 143.37 | 104 |
| warm | 112/112 | 96/112 | 96 | 13.46 | 21.10 | 0 |

All 96 definitive answers were correct against the record-derived reference.
Eight absent versions and eight unsupported questions correctly returned MAYBE;
they remain in the 112 denominator. The 112 queries are generated from one project
index, not 112 independently human-reviewed real-world questions. Versions and
formats are correlated. No population-wide statistical guarantee follows.

The protocol, source snapshot hash and 112 inputs were fixed before predictions.
Two preflight failures (sandbox direct networking and prerelease schema support)
scored zero cases; both are preserved. No benchmark cases were removed or relabeled.
Only synthetic unit fixtures were used to develop the tool before this evaluation.

## What changed

A local agent can retrieve primary release records, preserve exact version IDs,
compare actual calendar dates and day differences, compute integer comparisons,
and compose explicit negation/conjunction/disjunction. SQLite stores reusable
source snapshots, not Oracle-generated answers. Responses include exact source
spans, hashes, timestamps and calculation operands. Missing evidence and network
failures do not become NO. The tool is exposed through SDK, CLI and operator-enabled
REST/MCP, while the general verifier's calibration gate stays unchanged.

## Important source conflict finding

The supplementary 12 announcement checks initially used a heading extractor that
found dates on only one page. Eleven missing headings were initially represented
as false matches; that must not be interpreted as 11 contradictions. The original
output is retained, and source-audit.json distinguishes unknown fields correctly.
Four pages have a Last Updated date different from the index date. Last Updated
is not necessarily the original release date. This illustrates why the tool's
record-conditional scope is essential. No disagreements were used to change gold.
A source hash proves integrity, not truth; independent review is still required.

## What has and has not been established

Cold means an empty local SQLite evidence cache; upstream/proxy caches were not
controlled. Warm operation made zero source requests and reused the same evidence
for related, nonidentical propositions. All cold downloads are included in local
SDK latency; API/MCP transport latency is not included. Supported-only timing is
reported separately so the eight unsupported fast paths do not hide latency.

No paid API, Gemini call, cloud compute or model weights were used. Local host,
bandwidth and electricity costs remain unmeasured. No new claim of monetary or
latency advantage over Gemini follows because a paired Gemini comparison was not
run on this new dataset. The prior general benchmark results remain unchanged.
NNUE, larger neural training and learned search are not promoted by these results.

This is a useful deterministic adapter, not a 95% general search engine. The next
quality gate is human-established labels on a broader, source-disjoint set of
release facts and difficult prose claims, with independent citation review and
both overall accuracy and coverage evaluated. No public release/OPS changes.

Verification: 245 tests pass on both Python 3.11 and 3.12 (one optional Torch module skipped). Lint and isolated wheel build pass. Real CLI retrieval also returned the expected source-scoped proof. No paid cloud resources were created or left running.
