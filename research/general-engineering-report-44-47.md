# General verification implementation and first optimization campaign

The 95% target remains unmet. This campaign implements the public general-claim
path and acceptance machinery; it does not complete the independent end-to-end
evaluation, qualification or release phases.

## Implemented behavior

The public verifier can preserve exact prose propositions, evaluate qualified
source passages using a pinned local model and group evidence before acceptance.
It supports explicit outer negation without paraphrase-based answer reuse. Search
and full acquisition use existing self-hosted adapters and bounded budgets.
Machine acquisition is not human review. Exact spans, freshness, contradiction
search, failure reasons and per-stage timings are retained. Legacy deterministic
serving remains available without the semantic model.

A discovered long-document starvation bug was corrected with lexical passage
ranking and round-robin document coverage. Required counter-evidence must be
acquired and evaluated before acceptance. Revocation, cancellation and deadline
checks also apply to cached results. Changing model/aggregator/request policy
invalidates qualification. Latest controller policy is v4; frozen v3 research
calibration remains archival and is not silently rebound to the new policy.

## Measured results

| Experiment | Result | Decision |
|---|---|---|
| Public API integration,207 old development cases |176/207 correct diagnostic candidates(85.02%); accepted definitive coverage0% | Keep integration; do not claim accepted accuracy improvement |
| Uncertainty-preserving aggregation | Same 176 correct;143 distinct definitive scores instead of 2; diagnostic Brier0.150738→0.102550 | Keep score-resolution fix; not a truth-probability guarantee |
| CPU adaptation,4096 training +500 newdevelopment cases | Baseline394/500(78.8%), adapted389/500(77.8%);512 updates,407.61 training seconds | Reject candidate; stable weights retained |
| Primary research calibration risk panel |223/244 selected groups correct; conservative lower bound86.76% | Pass 80% research risk check only |
| Separate 500 case validation | Raw candidates391/500(78.2%); research-selected precision226/244(92.62%), coverage48.8%, overall63.8% | Does not meet 95%accuracy/coverage gates |

Accepted serving decisions still abstain without audited policy qualification.
Research mapping estimates are not objective truth probabilities or universal
future guarantees. Calibration improves how uncertainty is measured; it does not
repair the underlying model's refutation errors. The failed adaptation lost eight
correct NO cases, one correct YES case and gained four correct MAYBE cases.

## Data and audit readiness

A new candidate pool reserves 500 development,500 validation,1000 calibration and
1000 locked passage cases, plus 7816 training candidates. Source/event components
are separated and prior research/training sources excluded. Existing campaign-39 metadata
was read only to exclude related sources; its labels were not scored. New locked
labels were not scored or used for training. Publisher human annotations remain
unchanged. Source/citation/domain audits are pending, and the separate end-to-end
world-fact dataset is not yet qualified. Passage NEI cannot substitute for web
truth. The qualification CLI reports blocked rather than fabricating a sealed set.

Source-specific Wikipedia/CC-BY-SA terms were inspected for private research.
No dataset or trained model public redistribution is authorized by that review.
FEVER source-component preparation failures and the empty-identifier repair are
preserved in the qualification directory. No source family was chosen based on
model evaluation outcomes.

## Failures, reproducibility and resources

An exporter path bug overwrote original campaign-44 per-case traces. Original aggregates
remain. Actual second-run traces were moved to campaign 45; archived campaign-44 source was hash
checked and reproduced separately. Reproduced timings are new observations,
not recovered original timings. The incident and all available records remain.

The temporary 8-vCPU EC2 worker was terminated and its role, profile and security
group deleted after collection. Price verified:$0.357/hour. Application runtime
was 501.57 seconds, excluding bootstrap; invoices and complete operational cost
remain unknown. 12 USD was reserved, not claimed as actual spending. AWS campaign
reservations now 84USD/unreserved 216USD; GCP 26USD/unreserved 24USD unchanged.
No paid-model calls, public endpoint, OPS change or public publication occurred.

The source pool, trained safetensors adapter, fitting/risk/validation records,
training curve, calibration maps, reliability plot and per-case metadata are
retained. The strongest previously validated baseline checkpoint is unchanged.

## Next acceptance gates

Complete independent source/citation/domain audits and end-to-end labels. Improve
refutation and multi-hop evidence understanding on development cases. Freeze the
serving policy, validate both tracks at 97%, recalibrate that exact policy and only
then open the two 1000 case tests. Both must reach 950 correct, with precision,
answerable coverage and citation correctness at 95% and zero critical regressions.

## Verification

272 tests passed on Python 3.11 and 3.12 (one optional dependency test skipped
on each core environment). Ruff, whitespace checks and isolated wheel build pass.
The real adapter training and reload ran in the separate training environment.
