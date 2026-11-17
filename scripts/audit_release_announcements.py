"""Audit cached announcement date fields without changing benchmark labels."""

import json
import re
from datetime import datetime
from pathlib import Path


def main():
    root = Path("research/experiments/campaign-43")
    source = json.loads((root / "screen.json").read_text())
    rows = []
    for original in source["source_crosschecks"]:
        version = original["version"]
        path = Path("/workspace/houndoracle-data/campaign-43") / f"announcement-{version}.html"
        if not path.exists():
            rows.append(original | {"heading_date_match": None, "note": "Unavailable page"})
            continue
        html = path.read_text()
        headings = re.findall(r'<h2 id="(\d{4}-\d{2}-\d{2})-version-', html)
        updated = re.search(r"<dt>Last Updated</dt><dd>([^<]+)</dd>", html)
        updated_date = (
            datetime.strptime(updated[1], "%b %d, %Y").date().isoformat() if updated else None
        )
        rows.append(
            {k: v for k, v in original.items() if k not in ["match", "dates"]}
            | {
                "explicit_release_heading_dates": headings,
                "heading_date_match": original["index_date"] in headings if headings else None,
                "last_updated_date": updated_date,
                "last_updated_matches_index": updated_date == original["index_date"]
                if updated_date
                else None,
                "note": "Last Updated is not independently established original-release time; missing heading is unknown, not a contradiction.",
            }
        )
    result = {
        "rows": rows,
        "label_changes": 0,
        "excluded_cases": 0,
        "independent_human_audits": 0,
        "interpretation": "Initial narrow heading extractor returned no dates on11pages. Its false match values must not be read as11factual contradictions. Source record proof remains conditional onindex; date meanings need care. LastUpdated differs from index in some legacyrecords, preventing unrestricted actual-release-date claims.",
    }
    (root / "source-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "explicit_heading_matches": sum(r.get("heading_date_match") is True for r in rows),
                "unknown_heading": sum(r.get("heading_date_match") is None for r in rows),
                "last_updated_differs": sum(
                    r.get("last_updated_matches_index") is False for r in rows
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
