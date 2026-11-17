# FP32 transformer-fusion screen — rejected

Hypothesis: offline attention, BiasGelu and layer-normalization fusion can reduce the native encoder bottleneck while preserving the selected campaign09 weights, head and calibration. Official ONNX Runtime BERT-compatible optimizer used12heads/768hidden with no approximate GELU or embedding normalization fusion. Protocol and artifact hashes are retained alongside this report.

Paired207 reused development queries, three repetitions, warm one-thread models: source P50/P95 53.05/112.48ms; fused54.69/117.43ms. Maximum absolute logit difference6.68e-6, identical predicted labels. Parity passed, but the predeclared15% material speed gate failed. Negative point reductions are not evidence of a statistically significant regression. Reject the fusion artifact; no locked evaluation or production promotion. Local weights remain outside the repository.

The source runtime already performs native optimizations. These measurements do not justify adding C++ wrappers or a second inference engine. No paid cloud execution, full-agent latency, GPU, citation or all-in cost result is inferred. See `benchmarks/results/campaign-13/screen.json` and paired records.
