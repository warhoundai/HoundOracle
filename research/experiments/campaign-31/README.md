# Narrow evidence context, research only

A controlled paired development experiment on 207 historical claim/evidence queries
compared concatenating the top three retrieved passages with selecting the first.
The corpus contains the supplied human evidence and other queries' passages;
this is not live web retrieval or a fresh independent test.

| Metric | Top three | Top one |
|---|---:|---:|
| Raw classifier accuracy | 65.22% | 77.78% |
| Raw definitive coverage | 75.36% | 67.63% |
| Correct definitive / all queries | 50.24% | 55.56% |
| Raw accepted error | 33.33% | 17.86% |
| Model/packing P50 | 144.95 ms | 63.42 ms |
| Model/packing P95 | 253.85 ms | 144.46 ms |

Paired source-group bootstrap accuracy gain interval: 7.25–17.87 percentage
points. This exploratory reused-development interval does not qualify a selected
production policy. Timings use alternating execution order, fixed warmup and one
paired pass per query. They exclude network retrieval and the agent workflow.

Keep the scoped research adapter, not a production verdict change. It requires
an explicit single-document scope, retains other evidence IDs, and marks every
candidate unqualified. General contradiction and multi-hop checks cannot be
replaced by taking the first passage. The same diagnostic public API run produced
zero definitive answers on this general-claim cohort. Recovery from context
contamination does not establish improvement over the best supplied-evidence
classifier, 99% accuracy, or savings against Gemini.

Reproduce with `benchmarks/retrieval_selection.py`; exact configuration, per-query
metadata, intervals and scope are preserved in `result.json`. Source passages and
model weights remain outside the Apache source repository.
