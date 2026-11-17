# Campaign 42: evidence audit pilot results

The requested target is 95% overall verdict accuracy. This is a paired
24-question development diagnostic using historical supplied evidence,
not an independent evaluation or live-search benchmark. No labels changed.

| System | Correct / 24 | Definitive coverage | Correct definitive / 24 | P50 seconds | P95 seconds | Model tokens USD |
|---|---:|---:|---:|---:|---:|---:|
| Gemini HIGH baseline | 17/24 (70.83%) | 62.50% | 11/24 | 5.22414728199999 | 53.68976861709997 | unknown (provider failures) |
| Gemini HIGH relation_audit | 18/24 (75.00%) | 58.33% | 11/24 | 7.310744675999985 | 31.962038541600002 | 0.142094 |

Observed candidate 95% target met: **False**.
Preregistered development improvement gate passed: **False**.
Neither outcome establishes independent 95% correctness. At this sample size,
even 23 correct of 24 would have substantial uncertainty. Exact per-arm
intervals and paired bootstrap intervals are in screen.json. All planned
slots, including failures/deferred requests, are reconciled before scoring.

The historical local NLI reference is reported separately in screen.json:
cached component scores on identical questions, without new local timing.
Gemini-assisted verification is an optional research comparator, not evidence
of fully local superiority. The stable public Oracle configuration is unchanged.

Costs shown are reported usage at the pinned list prices, not invoices or
all-in operating cost. Host, storage, logging and electricity costs remain
unmeasured. $4 was reserved from the authorized $50 GCP campaign; unused
reservations are retained conservatively until invoices are reconciled.
No grounding or external searches were performed in this comparison.

No citation entailment audit, calibrated final-answer confidence, cloud-wide
billing guarantee, independent release qualification or 95% production claim
follows from this run. Source-label ambiguities require independent review;
they are not resolved by relabeling model disagreements. No locked evaluation
data was opened. Temporary-job cleanup is recorded in active-job.json.

Verification: 226 tests pass on Python 3.11 and 3.12, with the optional Torch
module skipped in core environments; lint and wheel build pass. Existing
Starlette deprecation warning remains. No OPS changes or public publication.

Two baseline requests returned provider 504 errors at 57.10 and 58.56 seconds.
They count as errors, not genuine MAYBE answers. Their token usage and charges
are unknown. The known baseline token subtotal is $0.096642 for 22/24 requests;
the complete baseline model cost is unknown. Candidate reported token usage totals
$0.14209425 for 24/24 requests. Comparing these as complete totals is invalid.
An initial accounting output that summed only known charges is preserved in
screen-before-cost-accounting-correction.json. The corrected screen marks the
baseline total unknown; accuracy, coverage and latency outcomes are unchanged.

The candidate fails the preregistered improvement gate and is not promoted.
Independent label/evidence review and a larger fresh comparison are required
before more prompt tuning can support a reliability claim. This bounded pilot
is closed; the temporary job and two previously completed campaign-14 jobs
have been deleted. No further model calls or retries were made.

Secondary label breakdown (8cases/class): audit YES7/8, NO4/8, MAYBE7/8; baseline YES5/8, NO6/8, MAYBE6/8. The prompt lost refutation performance. All four candidate NO disagreements have a blinded review packet; initial inspection also found possible annotation/scope conflicts. None was relabeled or removed. Label-review.json records pending review, not an independent quality finding. More repeated prompt tuning on these cases cannot establish general95% reliability.
