# Reproduction and artifact map

The stable checkpoint is `9123fdb`. The new research implementation is isolated in
`houndoracle.experiments`; the production verifier only gains diagnostics.

1. Sync benchmark/dev/API/MCP/S3 extras with the committed `uv.lock`.
2. `uv run python scripts/prepare_campaign_data.py --source /workspace/houndoracle-data/fever-sources --output NEW_CAMPAIGN_DIRECTORY`
3. `uv run python scripts/prepare_nli.py --directory /workspace/houndoracle-models/minilm`
4. Run `scripts/run_nli_scores.py` for fp32 and int8 on development/validation/calibration; the script never opens evaluation and refuses incomplete existing traces.
5. Run `scripts/analyze_campaign.py --scores ... --output ... --variant fp32 --train-aggregators`; then `scripts/score_heads.py` for both encoder variants and analyze each head separately.
6. Seal the untouched evaluation, freeze artifacts, and run `scripts/final_evaluation.py` once. Paths in the supplied campaign scripts identify this workspace snapshot; change them for a new campaign, never overwrite this holdout.
7. `scripts/profile_reuse.py` measures repeated validation latency; `scripts/plot_campaign.py` produces scientific plots. Re-running measurements is not permission to retune the locked test.

Recorded source hashes and final IDs: `experiments/dataset-manifest.json`.
Frozen model/calibration identities: `experiments/final-freeze.json`.
Encoder provenance: `model-comparisons/minilm-provenance.json`.
Trained heads and calibration mappings: `experiments/results/`.
Full questions, evidence, verdicts and per-query measurements:
`/workspace/houndoracle-data/locked-results/` (SHA hashes are in committed summary).
Committed aggregate and ID-level results: `../benchmarks/results/campaign-02/`.
Raw encoder/fit traces: `/workspace/houndoracle-data/scores/`.
Invalidated traces: `/workspace/houndoracle-data/scores-invalidated-v1/`.

The calibration artifacts preserve the original fit-time trace digests. Later head
trace citation/timing metadata was corrected without changing any score, label or
candidate used by calibration. No post-test refitting occurred. Reproduction should
compare the statistical tuples and learned parameters, rather than expecting wall
clock measurements to hash identically.

Actual Vertex generations succeeded in campaign 03. See
[the execution report](experiments/campaign-03/README.md), frozen cohort/code digests,
source-text-free task records and local analysis commands. The older Vertex training
job remains quota-blocked; it is not the execution path used for these results.
The $3 authorization is specific to the bounded matched-evidence pilot and does
not cover grounded runs. Re-execution after its declared resource envelope is not
automatically authorized.

The larger live follow-up is [campaign05](experiments/campaign-05/README.md).
Use its original frozen manifest/commit for inference reproduction, not the
later citation-guard implementation. Its bounded recovery preserved the original
modules and skipped all completed keys. `scripts/finalize_cloud_fresh.py` retains
all failed/unexecuted planned tasks; `scripts/compare_cloud_bounds.py` propagates
unknown usage and timers into conservative bounds. Do not spend against the
closed execution window or tune future models on these now-scored fresh questions.
