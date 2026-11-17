# Controlled local adapter training

Two real training arms used the same512human-labeled contrasting revision pairs,
1,024rows from512source components, seed1983and identical batch order. Median
pair word overlap92.86%; no selected pair had identical model input with different
labels. Training components do not overlap validation or calibration. Some rows
were already seen by the base model; this is continued supervision, not new
independent training data or an independently labeled evaluation.

Both arms trained100,611parameters: rank8query/value adapters in the last4attention
layers plus the classifier. Base-model weights remained frozen and unchanged. Initial
logit parity error was1.67e-6. Each arm completed128updates. CE-only took207.81s;
CEplus0.2pairwise softplus ranking took199.35s. Local CPU2threads, no new cloud
spending, small safe checkpoints retained privately with hashes in screen.json.

On the same207reused development queries:

|Configuration|Correct|Raw classifier accuracy|
|---|---:|---:|
|Frozen29base|176/207|85.02%|
|CE-only adapter|179/207|86.47%|
|Contrastive adapter|177/207|85.51%|

The CE-only gain was3corrected cases, no new errors on this panel; exploratory
paired95%gain interval0–3.38percentage points. It failed the frozen2-point material
gain gate and was not promoted. Ranking performed worse than the CEcontrol. The
small CE signal motivated a separately declared continuation experiment38, not a
retroactive gate change. No risk/test cohort was consumed and no calibration fit
was made for either changed model. These relation scores are not factual confidence.

Reproduce with prepare_contrastive_adapters.py and train_contrastive_adapters.py;
CPUlimit, inputs hash, initial checkpoint and exact configuration are in protocol.json.
Safe state restoration rejects a mismatched parameter schema. Unit tests verify
zero-initialization parity, frozen base gradients, objective direction and checkpoint
round-trip. Source text and weights remain outside the Apache source repository.
