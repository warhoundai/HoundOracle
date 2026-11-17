"""Record installed package declarations; this is not an automatic legal clearance."""

import argparse
import importlib.metadata
import json
from pathlib import Path


def inventory():
    rows = []
    for dist in importlib.metadata.distributions():
        metadata = dist.metadata
        classifiers = [x for x in metadata.get_all("Classifier", []) if x.startswith("License ::")]
        rows.append(
            {
                "name": metadata["Name"],
                "version": dist.version,
                "license_expression": metadata.get("License-Expression"),
                "license": metadata.get("License"),
                "classifiers": classifiers,
            }
        )
    return sorted(rows, key=lambda r: r["name"].lower())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(
            {
                "scope": "installed environment including development tools; declared licenses only",
                "packages": inventory(),
            },
            indent=2,
        )
        + "\n"
    )
