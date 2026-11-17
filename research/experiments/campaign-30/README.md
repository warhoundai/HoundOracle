# Full-pipeline retrieval gap diagnostic

The public API and research encoder are separate systems. On the 207 reused
historical development claims, the public engine answered zero definitively and
achieved30.92% all-decision accuracy by returning MAYBE. Its conservative compiler
reported unsupported/ambiguous general claims. This is a product coverage blocker;
research classifier accuracy must not be advertised as API accuracy.

A local corpus built from supplied evidence passages retrieved the gold document
at rank1 for97.2% of supported/refuted labels and within10 for99.3%. This constructed
corpus is not an independent open-web recall test. The raw research encoder scored
78.26% with supplied passages and65.22% after concatenating top3 hits. This motivated
campaign31's narrower context experiment, not more external search calls.

Source passages remain outside the repository. `retrieval-diagnostic.json` retains
per-query verdicts, failure reasons, ranks and timing. Reproduce with
`benchmarks/retrieval_gap.py` and the retained private input/checkpoint paths.
