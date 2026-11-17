# Campaign 42: evidence relation audit

User's revised pilot target: 95% overall decision accuracy, with coverage reported
separately. Compare baseline and experimental relation-audit prompts using Gemini
3.8 Flash HIGH, identical evidence and randomized paired order. Twenty-four
reused development questions, 48 calls maximum. This is a development diagnostic,
not independent qualification or a fully local system measurement.

The experimental prompt audits necessary factual relationships and distinguishes
missing information from explicit incompatible evidence. No question-specific
rules, labels or Oracle predictions are supplied to Gemini. The existing default
prompt and public Oracle configuration remain unchanged.

Protocol and source hashes were frozen before launch. Experimental runner:
`scripts/cloud_relation_runner.py`; reconciliation:
`scripts/assess_relation_pilot.py`. Original evidence and label files stay private
pending licensing/audit review; public outputs contain per-case metadata only.
No locked campaign-39 examples are consumed. Full citation entailment audit and
final correctness calibration remain outstanding.

Spending reservation: $4 from existing $50 GCP authorization, including at most
$1.032192 model-token reservations, $0.50 compute allowance and contingency.
No retries, grounding, GPUs, public endpoints, new IAM grants or always-on service.
Actual token list-price usage, host cost and invoices are distinct measurements.
The temporary job must be deleted after collecting all records.
