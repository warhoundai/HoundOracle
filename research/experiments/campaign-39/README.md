# Fresh evaluation sealed, not scored

Prepared1000new risk and500new locked source/event components before seeing38's
result. Connected pages, claims, passages and cases; excluded all previously used
components and prior source signatures. Hash selection did not filter labels or
predictions. Exact dataset hashes and source hashes are in dataset-manifest.json.

The continuation38failed its development gate, so neither cohort was scored.
Do not open these labels for model selection or bypass this candidate's guard.
The source files were mechanically imported to construct the sealed data; their
outcomes were not evaluated against a candidate. New snapshot URLs include oldid;
upstream passage matching and citation support still need independent review.

A gate-order regression test caught a new assessment bug before any fresh scoring:
the assessment checked the risk certificate after attempting to read locked results.
It now rejects before reading locked labels. Both that regression and changed-model
rejection pass. This does not invalidate previous experiments: the new cohorts were
not consumed. Scorers also verify source/model hashes and complete paired records.

The frozen future protocol specifies a dedicated correctness fit, a one-sided risk
gate, conservative paired noninferiority bounds and a separate strict99%goal. There
is no current passing candidate or public qualification. All source text stays private.
