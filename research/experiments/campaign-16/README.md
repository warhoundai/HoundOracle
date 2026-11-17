# Fixed checkpoint average — rejected

A single equal-weight FP32 parameter average of related checkpoints1920/1921 preserved the architecture, tokenizer and one-encoder-call cost. No gradient training or interpolation sweep. Nonfloating parameters were required identical; safe checkpoint hashes, four dynamic Torch/ONNX parity checks and exact parent hashes are retained.

The averaged encoder was actually run over3,201 head-training examples plus207 reused development queries. One equally supervised logistic head used the original five features, C1. Raw validation accuracy was77.29%, compared with77.78% for the continued checkpoint; correct definitive counts before calibration115 versus117. It failed the2point material-gain screen and is rejected. No fresh risk or locked evaluation was consumed and no production model changed.

This result rejects the fixed midpoint, not every possible checkpoint averaging method. No alpha tuning followed these results. Disk reclamation removed only previously rejected, reproducible runtime conversions; all parent/best trained checkpoints and manifests/profiles remain. No cloud charges by this local experiment.
