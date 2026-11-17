import argparse
import json

from houndoracle.experiments.dataset import build

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = build(args.source, args.output)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("provenance", "splits")}, indent=2)
    )
    print(
        json.dumps(
            {k: {a: b for a, b in v.items() if a != "ids"} for k, v in result["splits"].items()},
            indent=2,
        )
    )
