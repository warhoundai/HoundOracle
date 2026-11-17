# Final mixed-precision episode — rejected

Freeze campaign09's selected weights, head and correctness map. Leave the final1,2or3 transformer blocks in FP32, quantizing eligible earlier MatMul/Gather operators with per-channel QInt8. The classifier Gemm was already floating; this experiment localizes encoder sensitivity rather than claiming a classifier repair. No gradient training or threshold adjustment.

A new1,000-component risk partition was used for qualification. Development screening reused207 validation queries. The same600-query quantization locked test remains unscored; it was not used to choose precision masks. Label agreements were96.14%,95.65%,96.14%, below the predeclared98% gate. All three are rejected even where local timing/memory improved. Mixed3 measured P50/P95 36.86/73.09ms and approximately146.3MB on disk, but speed does not override quality.

Stop this bounded quantization episode. Keep FP32. Per-variant risk artifacts, profiles, source hashes and exact gates are retained. No NNUE, learned search, C++ or Gemini benefit is claimed. Public model weights/text are not included.
