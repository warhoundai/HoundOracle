# Larger generic NLI model — rejected

The initially considered ANLI-trained model was excluded after discovering
noncommercial upstream ANLI data terms; an MIT model card alone did not resolve
that restriction. The separate licensing record is retained.

A bounded AWS CPU worker evaluated `cross-encoder/nli-deberta-v3-base` at the
pinned revision and file hashes in `screen.json`. On the same 207 development
questions and supplied passages, raw accuracy was 63.29%, versus 77.78% for the
stable smaller model/head. Reject; no independent risk or locked test was opened.

Model acquisition took 9.93 seconds, loading 1.67 seconds and the measured worker
program 25.64 seconds. Peak RSS was 1,402,044 KiB. These timings are not a paired
hardware comparison or full agent latency. The EC2 instance terminated and its
security group was deleted. Actual invoiced cost is unknown; the $5 reservation
is not reported as expenditure. Upstream dataset/model rights remain separate
from this Apache code; no source passages or weights are published here.
