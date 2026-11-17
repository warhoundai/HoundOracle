"""Freeze qualified query-population models before reading new locked predictions."""

import hashlib
import json
from pathlib import Path

from fit_query_risk import SPECS


def main():
    root = Path("research/experiments/campaign-09")
    target = root / "locked-authorization.json"
    if target.exists():
        raise FileExistsError("Query selection already frozen")
    reports = {n: json.loads((root / (n + "-analysis.json")).read_text()) for n in SPECS}
    eligible = [n for n, r in reports.items() if r["certificate"]["qualified"] is True]
    selected = (
        max(
            eligible,
            key=lambda n: (
                reports[n]["calibrated_validation"]["correct_definitive_fraction"],
                reports[n]["calibrated_validation"]["accuracy_all_decisions"],
                reports[n]["raw_validation"]["accuracy_all_decisions"],
            ),
        )
        if eligible
        else None
    )
    files = ["dataset-manifest.json"] + [
        n + suffix
        for n in SPECS
        for suffix in ["-head.json", "-calibration.json", "-analysis.json"]
    ]
    output = {
        "qualified": selected is not None,
        "model_selected": selected,
        "models": {
            n: {
                "risk_qualified": r["certificate"]["qualified"],
                "correct_definitive_validation": r["calibrated_validation"][
                    "correct_definitive_fraction"
                ],
            }
            for n, r in reports.items()
        },
        "file_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files
        },
        "locked_predictions_seen": False,
        "production_qualified": False,
        "population": "Component-uniform one query per independent revision component; no all-pair or current-world guarantee.",
    }
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
