"""Render actual reconciled pilot measurements; unknown costs remain unknown."""

import json
from pathlib import Path


def main():
    folder = Path("research/experiments/campaign-42")
    result = json.loads((folder / "screen.json").read_text())
    lines = [
        "# Campaign 42: evidence audit pilot results",
        "",
        "The requested target is 95% overall verdict accuracy. This is a paired",
        "24-question development diagnostic using historical supplied evidence,",
        "not an independent evaluation or live-search benchmark. No labels changed.",
        "",
        "| System | Correct / 24 | Definitive coverage | Correct definitive / 24 | P50 seconds | P95 seconds | Model tokens USD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in result["systems"].items():
        n = s["count"]
        cost = s["reported_token_list_price_usd"]
        cost_text = f"{cost:.6f}" if cost is not None else "unknown (provider failures)"
        lines.append(
            f"| Gemini HIGH {name} | {round(s['accuracy_all_decisions'] * n)}/{n} "
            f"({s['accuracy_all_decisions']:.2%}) | {s['definitive_coverage']:.2%} | "
            f"{s['correct_definitive_count']}/{n} | {s['latency_seconds']['p50']} | "
            f"{s['latency_seconds']['p95']} | {cost_text} |"
        )
    lines.extend(
        [
            "",
            f"Observed candidate 95% target met: **{result['observed_95pct_target_met']}**.",
            f"Preregistered development improvement gate passed: **{result['screen_passed']}**.",
            "Neither outcome establishes independent 95% correctness. At this sample size,",
            "even 23 correct of 24 would have substantial uncertainty. Exact per-arm",
            "intervals and paired bootstrap intervals are in screen.json. All planned",
            "slots, including failures/deferred requests, are reconciled before scoring.",
            "",
            "The historical local NLI reference is reported separately in screen.json:",
            "cached component scores on identical questions, without new local timing.",
            "Gemini-assisted verification is an optional research comparator, not evidence",
            "of fully local superiority. The stable public Oracle configuration is unchanged.",
            "",
            "Costs shown are reported usage at the pinned list prices, not invoices or",
            "all-in operating cost. Host, storage, logging and electricity costs remain",
            "unmeasured. $4 was reserved from the authorized $50 GCP campaign; unused",
            "reservations are retained conservatively until invoices are reconciled.",
            "No grounding or external searches were performed in this comparison.",
            "",
            "No citation entailment audit, calibrated final-answer confidence, cloud-wide",
            "billing guarantee, independent release qualification or 95% production claim",
            "follows from this run. Source-label ambiguities require independent review;",
            "they are not resolved by relabeling model disagreements. No locked evaluation",
            "data was opened. Temporary-job cleanup is recorded in active-job.json.",
            "",
            "Verification: 226 tests pass on Python 3.11 and 3.12, with the optional Torch",
            "module skipped in core environments; lint and wheel build pass. Existing",
            "Starlette deprecation warning remains. No OPS changes or public publication.",
        ]
    )
    (folder / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
