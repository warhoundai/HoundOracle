# Fully local deployment regression — 2026-11-16

The required API/MCP serving path uses local SQLite evidence and local computation. It does not instantiate hosted search providers, even when vendor credentials are present. Imports remain unreviewed; an unavailable or unqualified answer remains MAYBE. Building the image and obtaining source/model snapshots require connectivity; serving the prepared snapshot does not.

Three temporary Ubuntu 24.04 c7i.xlarge workers in us-east-1 tested the implementation. First two workers each passed 163 tests and reproduced all 600 frozen neural decisions inside an OS network namespace without outbound connectivity. These are previously evaluated historical queries, not a new independent accuracy benchmark. The final source adds a stop-word retrieval regression (164 tests locally).

The Docker image built successfully on EC2. Final container ran with `--network=none`, read-only root, UID 10001 and temporary writable data. Authenticated local search passed, unqualified verification abstained, remote search was rejected, and hosted search/model calls were zero. See final-container.json and the source archive hashes. No vendor credentials were bundled.

Two harness failures are retained: attempt 1 tried copying into a read-only container filesystem; attempt 2 expected a remote-search rejection for a query that actually matched the local word “in.” The final probe uses a unique missing term. Local discovery now filters common function words when meaningful terms exist; this is a regression fix, not a demonstrated factual-accuracy improvement.

At the existing research threshold, 222/600 answers were definitive, 203 correct: 91.44% selective precision, 37% coverage, 33.83% correct-definitive fraction. At the strict 0.99 confidence threshold, none qualified; the 99% precision gate FAILS. Citation entailment remains independently unaudited. Do not claim 99–100% accuracy or improved general accuracy from deployment fidelity.

All three instances terminated: i-0933e934b4b546515, i-0057d140b1bf859f7, i-065eafb6229653207. Temporary role/profile houndoracle-local22-20261116 and security group sg-045654c005eb69e43 were deleted; no attached test volumes remained. No public endpoint or SSH ingress was provisioned. Each attempt reserved $5 within the authorized $300 AWS campaign; quoted compute rate was $0.1785/hour plus storage/transfer. Actual invoice totals remain unknown and are not reported as zero. Private artifacts retain the existing short S3 lifecycle.

Next accuracy work needs independently reviewed evidence spans and representative local-corpus retrieval labels. Current historical supplied-evidence calibration cannot justify arbitrary current-fact claims. Keep the stable model; no NNUE/C++ model was promoted.
