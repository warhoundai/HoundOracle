"""Report the scope, measured results and limitations of the record-tool pilot."""

import hashlib
import json
import platform
import sqlite3
from pathlib import Path

from houndoracle.experiments.metrics import summary


def main():
    root = Path("research/experiments/campaign-43")
    result = json.loads((root / "screen.json").read_text())
    audit = json.loads((root / "source-audit.json").read_text())
    sources = [
        "src/houndoracle/tools/releases.py",
        "src/houndoracle/tools/official_transport.py",
        "scripts/benchmark_release_records.py",
    ]
    runtime = {
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        "cold_means": "Empty local evidence database per question; upstreamCDN/proxycache notcontrolled",
        "model_calls": 0,
        "new_cloud_spending_usd": 0,
        "host_and_bandwidth_cost_usd": None,
    }
    (root / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    table = []
    supporting = {}
    for mode, values in result["systems"].items():
        rows = [
            json.loads(l)
            for l in Path(f"/workspace/houndoracle-data/campaign-43/{mode}-records.jsonl")
            .read_text()
            .splitlines()
        ]
        supporting[mode] = summary([r for r in rows if r["expected"] in ["YES", "NO"]])
        table.append(
            f"| {mode} | {round(values['accuracy_all_decisions'] * values['count'])}/{values['count']} | {values['definitive_count']}/{values['count']} | {values['correct_definitive_count']} | {values['latency_seconds']['p50'] * 1000:.2f} | {values['latency_seconds']['p95'] * 1000:.2f} | {sum(r['requests'] for r in rows)} |"
        )
    (root / "supported-only.json").write_text(json.dumps(supporting, indent=2) + "\n")
    text = (
        """# Primary-record tool: measured pilot

The local tool matched all 112 mechanically derived reference answers. This exceeds
95% **for this source-scoped record-relation pilot only**. The original general
fact-verification benchmark has not improved to 95%; this new test must not replace
or be pooled with it. No production/general-confidence qualification is granted.

| Local SDK mode | Correct verdicts | Definitive coverage | Correct definitive | P50 ms | P95 ms | Source requests |
|---|---:|---:|---:|---:|---:|---:|
"""
        + "\n".join(table)
        + """

All96 definitive answers were correct against the record-derived reference.
Eight absent versions and eight unsupported questions correctly returned MAYBE;
they remain in the112 denominator. The112 queries are generated from one project
index, not112 independently human-reviewed real-world questions. Versions and
formats are correlated. No population-wide statistical guarantee follows.

The protocol, source snapshot hash and112inputs were fixed before predictions.
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
as false matches; that must not be interpreted as11 contradictions. The original
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
"""
    )
    (root / "report.md").write_text(text)
    print(
        json.dumps(
            {
                "point_target_scoped": result["target95_scoped_point_met"],
                "general95": False,
                "crosschecks_differing_last_updated": sum(
                    r.get("last_updated_matches_index") is False for r in audit["rows"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
