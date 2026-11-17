# Per-tensor quantization: faster but rejected

Two fixed U8U8 dynamic transformations of the selected continued encoder, same frozen head/map, validation207 and1,000 new independent risk queries. No gradient training or threshold changes. Official ONNX Runtime quantization documentation was reviewed; quantization is explicitly lossy.

| Variant | ONNX bytes | Local P50 | Local P95 | Validation label agreement | Result |
|---|---:|---:|---:|---:|---|
| FP32 |328,597,332|53.62ms|110.98ms|100%|Retain|
| Dense weights |199,517,307|34.36ms|73.38ms|93.24%|Reject|
| Dense + embeddings |82,521,759|34.48ms|75.59ms|94.69%|Reject|

Dense quantization lost1.93points raw validation accuracy and5.31points definitive coverage. Full quantization improved raw accuracy slightly but lost6.76points coverage. Both miss the fixed98% fidelity and coverage gates despite passing independent query-risk checks. No quantized model replaced FP32; no locked predictions were made.

Timings are isolated one-thread local `verify_evidence` calls, warm model, result cache disabled,207 identical queries ×3 repetitions per model. Peak process RSS was640MB /508MB /283MB respectively. These are process peaks including loading, not static weight sizes or end-to-end network/agent/Gemini measurements. Question-bootstrap intervals are recorded; they do not capture all host variability.

Follow-up campaign11 tests fixed per-channel QInt8 variants. The host advertises AVX512_VNNI, which official ORT documentation states avoids the cited U8S8 saturation issue. It preserves all quality gates. The reserved600-query snapshot stays untouched and may be used only after new selection is frozen before first prediction. No cloud expenditure or public redistribution.
