# Complementary encoder stacking — rejected

Fixed standardized logistic stacking combined the stable and scaled-training
encoders, disagreement features and evidence metadata. Training, validation and
calibration-fit source groups were separated. The development panel was reused;
no fresh risk or locked evaluation data were opened.

Raw accuracy fell from 77.78% to 77.29%. At the provisional 80% policy, correct
definitive answers fell from 78 to 76 of 207, while coverage fell from 41.06% to
38.65%. Higher conditional precision did not justify lower useful-answer yield
and a failed coverage gate. Reject before independent evaluation; preserve the
stable checkpoint. These fitted scores are not qualified confidence estimates.

See `protocol.json`, `screen.json`, `head.json` and `mapping.json` for the exact
frozen configuration and outcome. Run `scripts/screen_dual_encoder.py` only with
the separately retained licensed private evidence inputs.
