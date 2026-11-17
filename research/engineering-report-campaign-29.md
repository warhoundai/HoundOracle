# HoundOracle continuation: independently measured semantic improvement

The strongest tested **research quality/useful-answer configuration** is now the
fixed DeBERTa adaptation in campaign29. It improves on stable1921 on a fresh sealed
600-query historical supplied-evidence cohort. It does not meet99%, qualify the
general public API, establish live-search performance, or justify public release.

## Measured result

| Fixed80% policy on same sealed600 | Stable1921 | Adapted DeBERTa |
|---|---:|---:|
| Overall verdict accuracy, including correct MAYBE | 60.17% | 76.50% |
| Definitive coverage | 36.33% | 53.67% |
| Correct definitive / all questions | 33.33% | 49.67% |
| Accepted precision | 91.74% | 92.55% |

The paired95% source-component bootstrap interval for overall accuracy gain is
12.50–20.33percentage points; for coverage gain13.33–21.50points. Raw candidate
accuracy is84.0%; raw outputs are not calibrated accepted answers. A preceding
untouched1000-component risk cohort passed the frozen80% conditional research gate
with491/514correct accepted and a93.36% one-sided lower bound. The model/map/policy
were not changed after opening these cohorts. Future tuning must use other data.

Training used6400human-labeled revision/FEVER replay examples, one800-update epoch,
AdamW2e-5 and frozen input embeddings. It took1092.16seconds on a temporary AWS CPU
worker. The actual safe checkpoint is retained locally with verified hashes.
The underlying model licenses and dataset terms remain separate from Apache code;
an ANLI-derived candidate was excluded for upstream noncommercial restrictions.

## Product questions

1. **Cheaper than Gemini?** Not established for this candidate. No new all-in paired
   Gemini measurement exists. Local inference avoids generative-model tokens but
   requires compute and has substantial abstention. Invoice costs remain unknown.
2. **Faster?** No new matched workflow P50/P95 advantage established. Local PyTorch
   model observations were277.32/496.02ms on the sealed run, excluding retrieval and
   agent orchestration. The separate context-packing experiment reduced model/packing
   P50/P95 from144.95/253.85ms to63.42/144.46ms, recovering accuracy lost by concatenating
   unrelated passages; it does not establish whole-agent or Gemini speed savings.
3. **More accurate?** Better than the stable local research reference on this cohort,
   by16.33points under the frozen80% verdict policy. No new same-query Gemini result;
   the previous live comparison favored Gemini over the prior calibrated model.
4. **Definitive and correct?**298/600,49.67% of all sealed questions. Coverage53.67%;
   accepted precision92.55%. At99% threshold, zero accepted answers: target failed.
5. **Central reuse economics?** Unmeasured. These tests use supplied historical
   evidence, not an S3 cache-hit savings or revalidation study.
6. **What helped?** Larger-model supervised domain adaptation improved independently
   measured correctness/useful-answer yield. Single-document input selection repaired
   context contamination in a separate constructed-corpus development test. Dual-model
   stacking and generic untuned larger NLI failed their gates and were rejected.
7. **NNUE/neural search/C++?** No demonstrated advantage or new production adoption.
   Semantic correctness and API scope remain bottlenecks. Existing native kernels
   already execute core inference; a wrapper rewrite is not justified by these results.
8. **Public release justified?** Not as a broad reliable verification engine. The
   general-claim public API diagnostic had zero definitive coverage. The conservative
   compiler, live retrieval, grounding/extraction, independent citation audit,
   broad/current/adversarial coverage and all-in Gemini comparison remain gates.

## Retention, controls and limits

All192Python tests passed on3.11/3.12; repository lint and wheel build passed.
Five new tests guard research scope and the joint99% gate. The prior stable checkpoint
and public confidence behavior remain intact. The scoped research retrieval adapter
always marks candidate verdicts unqualified and preserves other document IDs.

The training instance terminated; its task-only IAM role/profile/policy/security
group were deleted and attached volumes were zero. No paid compute remains from
these experiments. AWS campaign reservations total47USD within300USD authorization;
unreserved253USD. GCP reservations18USD within50USD; unreserved32USD. These values
are reservations, not actual expenditure or hard account-wide caps. Private S3
artifacts use the existing expiring bucket; the retained checkpoint is local.
No public repository push, OPS change or existing-product deployment occurred.

The live comparison is genuinely blocked: read-only inspection of the specified
GCP project returns requires reauthentication. Reconnection has been requested;
spending authorization is already present. Independent source-span review and
broader full-pipeline evidence are also absent. Repeatedly tuning against the newly
consumed sealed cohort would invalidate the experiment, not achieve99%.

Use this configuration for research on permissioned historical supplied evidence
inside the declared population. Retain abstention and source provenance. A general
Gemini replacement and a universal99–100% claim are unsupported. Next experiments
should evaluate grounded general-claim compilation/extraction and the frozen new
candidate on fresh live tasks with matched Gemini evidence and explicit citation
adjudication, rather than more tuning on these outcomes.

Artifacts: [campaign29](experiments/campaign-29/README.md),
[best research configuration](experiments/campaign-29/best-research-configuration.json),
[leaderboard](performance-leaderboard.md), `benchmarks/results/campaign-29/`,
[context experiment](experiments/campaign-31/README.md).
