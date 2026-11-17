# Larger public-data training run — completed, candidate rejected

A real additional CPU fine-tuning run processed9,601 examples in1,201 optimizer steps:6,401 unused human-labeled real VitaminC revision pairs,1,600 earlier revision replay pairs and1,600 FEVER replay examples. The global page/case/exact claim/span graph excluded all previously selected components from fresh training, risk and test selections. The1,000 fresh risk and600 fresh test queries were sealed before training and never sent to the worker or scored after the failed development gate. No Gemini-derived training labels.

Warm-start1921, one fixed epoch, AdamW2e-5/.01weight decay, batch8, gradient clipping1,256tokens, seed1935, optimizer reset. Actual SGD time924.334seconds on one8vCPU c7i.2xlarge. Safe checkpoint b9c422ab6bf63dc35d92ceede86a45332e02e30bd319ec84aa5959e1e9d09952. Export parity and downloaded hashes passed; source/data/configuration and all training losses are retained. This is actual training, not a calibration-only change or a global optimum search.

## Development gate

A new logistic head used exactly the same3,201 prior head-training examples, standardized five features and C1 as the stronger frozen control. Current encoder raw development accuracy77.78%; trained candidate76.81% (−0.97points). Candidate's validation-Brier-selected isotonic map reduced provisional coverage41.06%→33.33%, and correct definitive/all37.68%→32.37%. Selected error also fell8.24%→2.90%, but sacrificing that many correct answers fails the declared material-improvement/coverage gates.

**Reject**. Preserve the validated1921 checkpoint. No new risk/locked data consumed, no default model changed and no production confidence claimed. This does not prove more data is always harmful or establish a significant accuracy regression; this exact additional epoch failed to demonstrate the required benefit. The experiment does not isolate data volume from additional gradient updates.

## Resource accounting and cleanup

AWS Pricing API verified0.357USD/hour for Linux shared c7i.2xlarge inus-east-1; CPU quota8 and no other running/pending instances were observed before launch. A10USD reservation was made inside the user's300USD ceiling. This is a reservation, not an actual invoice. Final compute/storage/request/transfer/IPv4 charges and credit balance remain unverified; training/development costs are separate from per-query economics.

The instance automatically terminated after completion. Its task-only role/profile and no-ingress group were deleted. Root volume was encrypted and marked delete-on-termination; IMDSv2 and shutdown-terminate behavior were verified. The isolated SSE-S3 bucket blocks public access and expires objects after2days; temporarily retained artifacts support the pending actual hybrid benchmark. Existing product infrastructure/buckets remain untouched. Detailed confirmations are in cleanup.json and aws-preflight.json.

Source passages, tokenizer inputs and weights remain private/outside the Apache source tree, subject to existing VitaminC/Wikipedia/FEVER/model terms. Derived manifests/outcomes do not relicense upstream data. No public model redistribution or OPS/GitHub publication occurred.

## Continue from measured evidence

The current bottleneck is calibrated correct-definitive coverage, particularly evidence contradiction/insufficiency and numeric claims. Blind additional epochs, larger heads, quantization or C++ do not have demonstrated value here. The next useful measurements are the actual frozen hybrid workflow and broader independently reviewed current/technical evidence; GCPaccess has expired and reconnection remains pending. Budget permission persists. Do not repeatedly tune against these reused development queries or consume heldout labels to rescue this failed candidate.
