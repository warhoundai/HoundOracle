# Supervised larger-encoder adaptation

One fixed CPU training epoch adapted the pinned DeBERTa-v3-base encoder to6,400
human-labeled revision/FEVER replay examples. Input embeddings were frozen; the
remaining86,043,651 parameters used AdamW2e-5, batch8, seed1963. Training finished
800updates in1,092.16seconds. Initial PyTorch/ONNX parity error was1.46e-5; the
classifier weights changed and the safe checkpoint hash was verified after local
retention. This is real training, not a simulated comparison.

## Development gate

| Metric on reused207-query panel | Stable MiniLM/head | Adapted DeBERTa |
|---|---:|---:|
| Raw classifier accuracy | 77.78% | 85.02% |
| Provisional80 correct definitive answers | 78 | 103 |
| Provisional80 definitive coverage | 41.06% | 52.66% |
| Provisional80 accepted error | 8.24% | 5.50% |
| Provisional99 definitive coverage | Not the selection baseline | 0% |

The exploratory paired raw accuracy gain interval is2.42–12.08percentage points.
This reused-development screen justified independent assessment, not production
promotion. The candidate has not reached99%; thresholding at99% accepts nothing.
Recorded host timings are from separate systems and cannot support a speed claim.

## Independent sequence

`independent-protocol.json` freezes the checkpoint, correctness map and80% policy
before opening campaign19's sealed1,000-component risk cohort. No risk component
appears in training, validation or calibration-fit (`risk-disjointness.json`).
The declared historical-population gate uses a one-sided Clopper–Pearson lower
bound at alpha.025 and at least30accepted groups. Only a passing risk gate permits
opening the sealed600-query cohort; no refit or threshold sweep is allowed there.
The joint99% goal additionally checks overall accuracy, selective precision,
coverage and independent citation review. It cannot pass without that review.

Reproduce training with `scripts/train_deberta_revision.py`; fixed offline cohort
scoring with `scripts/score_deberta_followup.py`; stage assessment with
`scripts/assess_deberta_followup.py`. Inputs, weights and quoted evidence remain
outside the Apache source repository. `input-manifest.json` and retained checkpoint
seals preserve provenance. Source model/data licenses and attribution are separate
from the software license. Public pretrained-model corpus contamination remains
possible even with our supervised split isolation.

## Resources and retention

One c7i.2xlarge worker, encrypted disposable disk, no ingress/public application,
least-privilege temporary S3 role,40minute shutdown watchdog and$5 application
reservation. The instance terminated, temporary role/profile/policy/security group
were deleted and no attached volumes remained. Actual invoiced cost is unknown;
the reservation is not actual spending. The checkpoint is retained locally;
three reproducible earlier ONNX exports were removed for disk space while retaining
their original safe weights and the complete stable1921checkpoint/export.
Default API confidence behavior remains unchanged. No OPS or public release.

## Independent risk outcome

The frozen80% policy passed its conditional research gate:491/514accepted answers
correct (95.53% empirical precision),51.4% definitive coverage,49.1% correct
definitive/all queries. One-sided component correctness lower bound93.36% at
alpha.025. Raw accuracy82.9%; selected all-decision accuracy78.0%. This is not
99% precision,99% overall accuracy or a source-citation audit. At99% the map
accepted zero queries; the strict goal failed. The1,000 source/event groups were
not used to refit the map or choose another threshold. The authorized sealed600
comparison was opened only after this passing risk gate.

Calibration Brier score0.09131 and10-bin ECE0.03873 describe eligible candidate
verdicts, not a guarantee. Descriptive curves exclude one truncated row and use
999 as denominator; full policy metrics retain all1,000. This distinction is
recorded in `risk-assessment-notes.json`.

## Sealed600 outcome — retain research candidate

| Fixed80% policy | Stable1921 | Adapted DeBERTa |
|---|---:|---:|
| Overall verdict accuracy, including MAYBE | 60.17% | 76.50% |
| Definitive coverage | 36.33% | 53.67% |
| Correct definitive answers | 200/600 | 298/600 |
| Accepted precision | 91.74% | 92.55% |

Paired95% source-component bootstrap intervals: overall accuracy gain12.50–20.33
percentage points; coverage gain13.33–21.50points. The candidate passed the frozen
research noninferiority margins. Raw neural accuracy was84.0%, with paired raw gain
interval4.83–11.50points. No retuning occurred after opening either fresh cohort.
The candidate is retained as the strongest tested research quality/useful-answer
configuration; the prior stable model remains available for rollback.

This does not meet99%. At the99% confidence threshold, coverage remains zero and
the strict joint goal fails. Citation support is unaudited. The broader public API
still has unsupported general claim compilation, and no default route was changed.
The local PyTorch P50/P95 observation was277.32/496.02ms; reference execution and
hardware/runtime paths were not a paired latency trial. No speed or cost advantage
against Gemini is claimed. New live Gemini testing is blocked by the connector's
read-only reauthentication failure. All1600candidate observations and600reference
observations are preserved without source quotations in `benchmarks/results/campaign-29/`.
The source/event IDs and URLs preserve provenance; quoted evidence stays private.
