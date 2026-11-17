# Campaign 48: freshness-aware retrieval

Hypothesis: applying freshness checks before the result limit recovers eligible
sources hidden behind expired top-ranked hits. Search must also reject evidence
whose upstream dependency has expired. Risk: additional SQLite work when many
hits are ineligible. Keep only if both regressions pass and legacy tests remain
passing; roll back if valid retrieval or provenance isolation regresses.

The old FTS query limited rows first and then filtered direct expiry. The new
query streams ranked rows, applies recursive dependency freshness, and stops
after enough eligible hits. Ties use document ID for deterministic ordering.
The serving policy fingerprint changes, invalidating old calibration/cache keys.

On 40 deterministic synthetic retrieval cases (four stale-document counts, ten
repetitions), baseline found the eligible record in 10/40; candidate in 40/40.
This is a targeted regression workload, not independent factual accuracy. Median
query time was 0.105 ms baseline and 0.238 ms candidate on this machine. The added
cost buys correct freshness handling; it is not a latency optimization. These
sequential microbenchmarks have no end-to-end latency or statistical superiority
claim. All observations are retained in benchmarks/results/campaign-48.

Keep the correctness fix. General accuracy remains unqualified. No model training,
locked evaluation, paid resources or public service deployment was performed.
