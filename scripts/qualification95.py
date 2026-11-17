"""Seal audited two-track cohorts and score only a frozen validation-qualified policy."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from houndoracle.experiments.qualification95 import TRACKS, assess_campaign, validate_partitions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["status", "seal", "score"])
    p.add_argument("--registry", type=Path, default=Path("research/qualification95"))
    p.add_argument("--partitions", type=Path)
    p.add_argument("--results", type=Path)
    p.add_argument("--validation", type=Path)
    p.add_argument("--policy-id")
    args = p.parse_args()
    sealed = args.registry / "sealed.json"
    if args.mode == "status":
        print(
            json.dumps(
                {
                    "status": "SEALED_AWAITING_VALIDATION"
                    if sealed.exists()
                    else "BLOCKED_AUDITED_PARTITIONS_NOT_SEALED",
                    "general95_qualified": False,
                    "required_tracks": sorted(TRACKS),
                    "locked_cases_per_track": 1000,
                }
            )
        )
        return
    if args.mode == "seal":
        if sealed.exists():
            raise FileExistsError("Preserve sealed cohort")
        if args.partitions is None:
            p.error("--partitions required")
        data = json.loads(args.partitions.read_text())
        manifest = validate_partitions(data)
        manifest["private_input_sha256"] = hashlib.sha256(args.partitions.read_bytes()).hexdigest()
        args.registry.mkdir(parents=True, exist_ok=True)
        with sealed.open("x") as f:
            f.write(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"sealed": True, "dataset_digest": manifest["dataset_digest"]}))
        return
    if not args.validation or not args.policy_id or not args.partitions or not args.results:
        p.error("score requires validation, policy-id, partitions and results")
    # Read validation first. Failed/missing gates must not open locked labels.
    validation = json.loads(args.validation.read_text())
    if validation.get("policy_id") != args.policy_id:
        raise PermissionError("Validation policy changed")
    accuracies = {}
    for track in TRACKS:
        entry = validation.get("tracks", {}).get(track, {})
        value = entry.get("accuracy")
        if (
            type(value) not in (float, int)
            or not math.isfinite(value)
            or not 0.97 <= value <= 1
            or entry.get("count", 0) < 500
        ):
            raise PermissionError(
                "Both97percentvalidationgates required before opening locked data"
            )
        accuracies[track] = value
    manifest = json.loads(sealed.read_text())
    for track in TRACKS:
        entry = validation["tracks"][track]
        expected = manifest["partitions"][f"{track}:validation"]
        if entry.get("dataset_sha256") != expected["sha256"] or entry["count"] != expected["count"]:
            raise PermissionError("Validation dataset changed")
    target = args.registry / "locked-result.json"
    if target.exists():
        raise FileExistsError("Preserve locked results; new qualification needs a fresh cohort")
    raw = args.partitions.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["private_input_sha256"]:
        raise ValueError("Sealed labels changed")
    data = json.loads(raw)
    outcomes = json.loads(args.results.read_text())
    result = assess_campaign(
        {t: data[t]["locked"] for t in TRACKS},
        outcomes,
        policy_id=args.policy_id,
        manifest=manifest,
        validation_accuracy=accuracies,
        critical_security_failures=validation.get("critical_security_failures", 0),
    )
    with target.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
