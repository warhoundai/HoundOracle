# Citation metric correction

The old `run_nli_scores.py`, `score_heads.py`, and `final_evaluation.py` populated `citation_supported` using candidate/decision agreement with the FEVER answer label. This is label agreement on supplied evidence; it is **not an independent citation-span quality assessment**. Existing frozen artifacts are preserved for audit. Their `citation_supported_definitive_fraction` values must not be presented as independently measured citation correctness.

New runs retain this diagnostic under `label_agreement_on_supplied_evidence` and leave `citation_supported` null. Aggregate citation support now requires every definitive row to record `citation_audit_method = independent_span_review` as well as an explicit support judgment. Missing or mixed review coverage yields null, not zero or an inflated percentage. Independent reviews still need to be performed and documented; this field is a provenance contract, not an automatic verifier of reviewer honesty.

Verdict labels, verdict accuracy, answer coverage and error rates are unaffected. Provider citation integrity checks (valid indices and exact reconstruction of supplied spans) are also distinct from independent semantic support assessment. No old citation scores have been silently rewritten.

The corrected campaign07 screen summary also suppresses inherited timing: original cached-logit times, head-only timings and measurements under concurrent CPU workloads do not establish a paired end-to-end speed comparison.
