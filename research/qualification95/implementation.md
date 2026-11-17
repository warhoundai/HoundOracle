# 95% campaign implementation status

The approved goal is unchanged: two separate 1,000-case tests, at least 95%
observed accuracy on each, plus precision, answerable coverage, citation and
security gates. No narrow tool benchmark substitutes for either task.

## Implemented

- Opt-in exact general-proposition compilation in the existing public verifier.
  Literal questions remain distinct; explicit outer negation is composed without
  rewriting arbitrary logical scope or using similarity for answer reuse.
- A pinned local safetensors evaluator, exact passage spans, dependency grouping,
  a grouped aggregator and per-stage traces. No model downloads, remote code or
  paid inference are triggered by the runtime.
- Full-document acquisition using the existing crawler/security boundary. Search
  snippets do not qualify; machine acquisition is separate from human review.
  Search, acquisition, deadlines, failures and contradiction investigation are
  bounded. Counter-evidence must be fetched before an online acceptance gate.
- Source content binding, freshness invalidation, policy-bound calibration and
  caches. Model, aggregator and request-budget changes invalidate qualification.
  Old research calibration cannot authorize this new serving policy.
- SDK/CLI activation and consistent REST/MCP behavior through Oracle.verify.
- Two-track partition validation, observed 95% scoring, denominator checks,
  citation audit requirements and the 97% validation guard before locked access.

## Data prepared, not qualified

A new VitaminC human-annotation pool has 500 development, 500 validation, 1,000
calibration and 1,000 locked passage candidates, plus 7,816 training candidates.
They have distinct source-connected components and exclude prior training and
research sources. Campaign-39 source metadata was inspected solely for exclusion;
its labels were not scored or used as training examples.

All new candidates retain pending source/audit/domain qualification fields. They
are not a sealed acceptance dataset. Dataset-specific license text permits
commercial reuse under the applicable Wikipedia/CC-BY-SA terms; attribution and
redistribution obligations remain separate. No public dataset/model release is
approved by this private training experiment.

FEVER train provenance and evidence-set checks were implemented and attempted.
The unused connected-component pool could not satisfy the class counts; that
failure remains recorded. A first overly broad exclusion and an empty-string
identifier parsing error are also preserved. There were no candidate predictions
used to choose the source family, exclude examples or change labels.

The end-to-end world-fact cohort still needs independently audited temporal/source
labels. Passage NOT ENOUGH INFO is explicitly rejected as automatic web ground
truth. The status command reports blocked until both audited tracks are sealed.

## Measurements and completed experiments

Campaign 44 runs all 207 old development cases through the public API. The new
path produces 176 correct diagnostic candidates (85.02%); accepted API coverage
remains zero without a new qualification artifact. This is integration progress,
not a production accuracy increase. Campaign 45 restores neural uncertainty:
143 distinct definitive scores instead of two; uncalibrated correctness-score
Brier changes from 0.150738 to 0.102550 with unchanged candidate verdicts.

An exporter path bug replaced the original campaign-44 per-case traces. Original
aggregates remain; the real second-run records were moved to campaign 45, and the
old source was hash-verified and reproduced separately. Reproduced timings are
new measurements, not reconstructed original timing observations.

Campaign 46 completed one bounded CPU adapter experiment on 4,096 training cases
and the separate 500-case development panel. Baseline accuracy was 394/500;
adaptation reduced it to 389/500, so the candidate was rejected. The worker was
terminated and its security group, IAM profile and role deleted. Only training
and development examples were uploaded. Stable weights remain unchanged.

Campaign 47 fitted a research-only calibration map on 500 cases, checked risk on
500 separate cases, then evaluated 500 validation candidates. Research selection
answered 244/500 definitively, with 226 correct (92.62% precision, 48.8% coverage).
Raw candidate accuracy was 391/500 (78.2%). These are not audited qualification
results. The artifact cannot authorize public decisions and binds archived policy
v3; latest v4 adds cache cancellation/freshness guards and requires recalibration.
Both new locked tests remain unscored. See the engineering report for limitations.

## Remaining gates

Complete independent source/citation audits and end-to-end labels; improve local
models and retrieval on development data; validate the fixed policy on both
tracks; fit final-pipeline calibration; only then open the new locked tests.
No 95% achievement, Gemini advantage or public-release readiness is claimed.
