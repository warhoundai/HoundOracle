import argparse
import json

from houndoracle.experiments.nli import download

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--directory", required=True)
    args = p.parse_args()
    print(json.dumps(download(args.directory), indent=2))
