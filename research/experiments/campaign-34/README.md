# Larger local generative verifier — rejected

After the4B architecture failed, this controlled follow-up changed model capacity
while retaining the prompt, JSON schema,48balanced development cases, evidence,
CPU inference library, seed, temperature and output limit. The Apache2 quantized
Qwen3-30B-A3B-Instruct-2507 checkpoint was hash-verified before loading. No hosted
inference, test-set tuning or additional training labels were used.

Result30/48correct(62.50%), versus adapted NLI35/48(72.92%) and prior4B32/48(66.67%).
Raw accepted error41.03%. Reject before independent evaluation. This run shows that
increasing capacity under this prompt/configuration did not repair the semantic
errors; it does not establish that every larger model or prompting method fails.

Measured model acquisition310.66seconds, program541.97seconds, inference P50/P95
4.285/5.723seconds, peak RSS31,136,648KiB. Hardware was an8vCPU/32GiB m7i.2xlarge;
reference NLI and4B timings were not a matched deployment trial. The verified
on-demand quote was.4032USD/hour; actual invoices remain unknown.15USDreservation
is an application ceiling, not reported expenditure or an account-wide hard cap.

The single worker terminated; encrypted root storage was delete-on-termination,
attached volumes were zero, and the task-only role/profile/policy/security group
were deleted. The private expiring S3 archive retains source inputs; public records
contain metadata only. Prior validated checkpoint and default API are unchanged.
No99%accuracy, citation-quality, general-web, all-in-cost or release gate passed.
