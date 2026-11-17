# Per-channel QInt8 episode — rejected

Hypothesis: per-channel signed quantization on the VNNI CPU preserves more semantic fidelity than per-tensor unsigned quantization. Freeze the selected campaign09 model/head/map. Compare dense-only and dense-plus-embedding quantization with fresh independent risk queries and the same development screen. No retraining or gate changes.

Warm local P50/P95 were18.13/37.63ms for dense-only and18.40/39.13ms for full quantization: roughly66% lower than FP32. Both changed3.86% of validation labels, failing98% label agreement, with accuracy/coverage losses beyond the declared gates. Both rejected. The600-query quantization locked test remained unscored. All timings are local research verifier timings, not end-to-end Gemini or agent measurements. Model files remain outside the Apache source tree.

Full configurations, independently fitted risk certificates, per-query profiles and the rejection disposition are preserved here and under `benchmarks/results/campaign-11/`.
