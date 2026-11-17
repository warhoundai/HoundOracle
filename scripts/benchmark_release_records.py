"""Source-scoped release tool benchmark. Not a general factual-accuracy claim."""

import hashlib
import json
import re
import time
from datetime import date, timedelta
from pathlib import Path

from houndoracle.experiments.metrics import summary
from houndoracle.tools.official_transport import fetch_official as fetch
from houndoracle.tools.releases import URL, ReleaseRecords


def main():
    private = Path("/workspace/houndoracle-data/campaign-43")
    public = Path("research/experiments/campaign-43")
    if (public / "screen.json").exists():
        raise FileExistsError("Evaluation already consumed")
    initial = fetch(
        URL, timeout=10, max_bytes=2_000_000, same_origin_redirects=True, accepted_statuses=(200,)
    )
    body = initial["body"]
    digest = hashlib.sha256(body).hexdigest()
    frozen = private / "index.json"
    if frozen.exists() and frozen.read_bytes() != body:
        raise ValueError("Frozen source changed before retry; no silent regeneration")
    frozen.write_bytes(body)
    # Reference uses the standard JSON array parser, independent of candidate span parser.
    data = json.loads(body)
    data.sort(key=lambda r: hashlib.sha256(("1993:" + r["version"]).encode()).hexdigest())
    cases = []
    for i, r in enumerate(data[:96]):
        v, d = r["version"], date.fromisoformat(r["date"])
        other = data[96 + i]
        w, e = other["version"], date.fromisoformat(other["date"])
        mode = i % 12
        year = d.year + int(i % 2 == 0)
        if mode == 0:
            q = f"Was Node.js {v} released on {d.isoformat()}?"
            expected = True
        elif mode == 1:
            q = f"Was Node.js {v} released on {(d + timedelta(days=1)).isoformat()}?"
            expected = False
        elif mode == 2:
            q = f"Was Node.js {v} released in {year}?"
            expected = d.year == year
        elif mode == 3:
            q = f"Was Node.js {v} released before {(d + timedelta(days=(-1 if i % 2 else 1))).isoformat()}?"
            expected = i % 2 == 0
        elif mode == 4:
            q = f"Was Node.js {v} released before Node.js {w}?"
            expected = d < e
        elif mode == 5:
            delta = (e - d).days
            threshold = max(0, delta - (1 if i % 2 else 0))
            q = f"Was Node.js {v} released more than {threshold} days before Node.js {w}?"
            expected = delta > threshold
        elif mode == 6:
            q = f"Did Node.js {v} include security fixes?"
            expected = r["security"]
        elif mode == 7:
            q = f"Is Node.js {v} marked LTS in the release index?"
            expected = r["lts"] is not False
        elif mode == 8:
            major = int(v[1:].split(".")[0])
            q = f"Does Node.js {v} have a major version greater than {major}?"
            expected = False
        elif mode == 9:
            q = f"Is it false that Node.js {v} not released in {d.year}?"
            expected = True
        elif mode == 10:
            q = f"Was Node.js {v} released in {d.year} and Was Node.js {w} released in {e.year + 1}?"
            expected = False
        else:
            q = f"Was Node.js {v} released in {d.year + 1} or Was Node.js {w} released in {e.year}?"
            expected = True
        cases.append(
            {
                "id": f"node43:{i}",
                "group_id": v,
                "question": q,
                "expected": "YES" if expected else "NO",
                "type": mode,
            }
        )
    for i in range(16):
        q = (
            f"Was Node.js v999.{i}.0 released in 2020?"
            if i < 8
            else f"Is Node.js v999.{i}.0 faster than every other runtime?"
        )
        cases.append(
            {
                "id": f"node43:{96 + i}",
                "group_id": f"unknown:{i}",
                "question": q,
                "expected": "MAYBE",
                "type": "missing" if i < 8 else "unsupported",
            }
        )
    (private / "inputs.json").write_text(json.dumps(cases, indent=2) + "\n")
    # Test inputs sealed before any candidate predictions.
    (public / "dataset-manifest.json").write_text(
        json.dumps(
            {
                "snapshot_sha256": digest,
                "cases_sha256": hashlib.sha256(
                    json.dumps(cases, sort_keys=True).encode()
                ).hexdigest(),
                "case_count": len(cases),
                "reference": "Mechanically derived records, no human gold or independent generalization claim",
            },
            indent=2,
        )
        + "\n"
    )
    runs = {}
    warm = ReleaseRecords(private / "warm.sqlite", transport=fetch)
    warm.refresh(allow_network=True)
    for mode in ["cold", "warm"]:
        output = []
        for case in cases:
            tool = ReleaseRecords(transport=fetch) if mode == "cold" else warm
            start = time.perf_counter()
            answer = tool.verify(case["question"], allow_network=mode == "cold")
            seconds = time.perf_counter() - start
            # Unsupported inputs need no network; known records must use the frozen index.
            if answer.get("snapshot_sha256") not in (None, digest):
                raise ValueError("Source changed: preserve failed run; do not rescore")
            record = (
                case
                | answer
                | {
                    "latency_seconds": seconds,
                    "all_in_cost_usd": None,
                    "failure": answer["reason"]
                    if answer["reason"]
                    in {
                        "RETRIEVAL_FAILED",
                        "SNAPSHOT_INTEGRITY_FAILED",
                        "INVALID_SNAPSHOT",
                        "STALE_EVIDENCE",
                    }
                    else None,
                }
            )
            output.append(record)
            if mode == "cold":
                tool.close()
        runs[mode] = output
    warm.close()
    audit = []
    for r in data[:12]:
        version = r["version"]
        url = f"https://nodejs.org/en/blog/release/{version}"
        try:
            response = fetch(
                url,
                timeout=10,
                max_bytes=2_000_000,
                same_origin_redirects=True,
                accepted_statuses=(200,),
            )
            html = response["body"].decode()
            matches = re.findall(r'<h2 id="(\d{4}-\d{2}-\d{2})-version-', html)
            (private / f"announcement-{version}.html").write_text(html)
            audit.append(
                {
                    "version": version,
                    "url": url,
                    "page_sha256": hashlib.sha256(response["body"]).hexdigest(),
                    "dates": matches,
                    "index_date": r["date"],
                    "match": r["date"] in matches,
                    "failure": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 -- retain inaccessible crosschecks
            audit.append(
                {"version": version, "url": url, "match": None, "failure": type(exc).__name__}
            )
    result = {
        "systems": {k: summary(v) for k, v in runs.items()},
        "target95_scoped_point_met": all(
            summary(v)["accuracy_all_decisions"] >= 0.95 for v in runs.values()
        ),
        "independent_general95_qualified": False,
        "production_qualified": False,
        "new_cloud_spend_usd": 0,
        "source_snapshot_sha256": digest,
        "source_crosschecks": audit,
        "limits": "Mechanically generated112questions from one official project index; shared facts/questions correlated. Source crosschecks share publisher. No binomial general-world guarantee, generative benchmark replacement, hidden abstentions or calibrated probability. All cold fetch and failed cases retained.",
    }
    (public / "screen.json").write_text(json.dumps(result, indent=2) + "\n")
    dest = Path("benchmarks/results/campaign-43")
    dest.mkdir(parents=True, exist_ok=True)
    for mode, values in runs.items():
        (private / f"{mode}-records.jsonl").write_text(
            "".join(json.dumps(v) + "\n" for v in values)
        )
        # Avoid distributing whole source excerpts before source licensing review.
        metadata = [
            v
            | {"citations": [{k: z for k, z in c.items() if k != "quote"} for c in v["citations"]]}
            for v in values
        ]
        (dest / f"{mode}-metadata.jsonl").write_text(
            "".join(json.dumps(v) + "\n" for v in metadata)
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
