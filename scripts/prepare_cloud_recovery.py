"""Resume only unattempted keys using the exact submitted inference modules and frozen policy."""

import ast
import json
import random
from pathlib import Path


def main():
    data = Path("/workspace/houndoracle-data")
    job = json.loads((data / "cloud-run-fresh.json").read_text())
    payload = json.loads((data / "cloud-fresh-payload.json").read_text())
    logs = json.loads((data / "cloud-05-initial-logs.json").read_text())
    queries = [
        r["jsonPayload"]["record"] for r in logs if r.get("jsonPayload", {}).get("kind") == "query"
    ]
    plan = []
    rng = random.Random(1907)
    for repeat in range(3):
        for row in payload["confirmation"]:
            arms = ["baseline", "selected"]
            rng.shuffle(arms)
            plan.extend({"id": row["id"], "repeat": repeat, "arm": arm} for arm in arms)
    for a, b in zip(queries, plan):
        if any(a[k] != b[k] for k in ["id", "repeat", "arm"]):
            raise ValueError("Logs are not the exact executed prefix")
    failed = plan[len(queries)]
    if failed["arm"] != "baseline" or failed["id"] != "fever:93342":
        raise ValueError("Failure location does not match observed traceback")
    old_attempts = sum(r["model_calls"] for r in queries) + 1
    if old_attempts != 179:
        raise ValueError("Recovery budget must be rebuilt for changed prior attempt count")
    pending = plan[len(queries) + 1 :]
    by_id = {r["id"]: r for r in payload["confirmation"]}
    failed_record = {
        **failed,
        "group_id": by_id[failed["id"]]["group_id"],
        "expected": by_id[failed["id"]]["expected"],
        "phase": "fresh",
        "decision": "MAYBE",
        "failure": "PROVIDER_DEADLINE_EXCEEDED",
        "model_calls": 1,
        "source": "provider_failure",
        "gemini": None,
        "oracle": None,
        "latency_seconds": None,
        "token_cost_usd": None,
        "reserved_usd": 0.021504,
        "latency_note": "Original runner crashed before emitting its per-query timer; no exact duration imputed.",
    }
    (data / "cloud-05-failure-record.json").write_text(json.dumps(failed_record, indent=2) + "\n")
    spec = job["spec"]["template"]["spec"]["template"]["spec"]
    original_code = spec["containers"][0]["args"][1]
    last = ast.parse(original_code).body[-1]
    original_runner = ast.literal_eval(last.value.args[0])
    helpers = original_runner[: original_runner.index("def main():")]
    counter = original_runner[
        original_runner.index("class FreshCounter:") : original_runner.rindex("if __name__")
    ]
    main_source = (Path(__file__).parent / "cloud_recovery_runner.py").read_text()
    runner = helpers + '\nRECOVERY=globals().get("RECOVERY",[])\n' + counter + "\n" + main_source
    boot = "".join(original_code.splitlines(keepends=True)[: last.lineno - 1])
    boot += (
        "exec("
        + repr(runner)
        + ', {"__name__":"__main__","PAYLOAD":PAYLOAD,"RECOVERY":'
        + repr(pending)
        + "})\n"
    )
    compile(boot, "<bounded-recovery>", "exec")
    spec["containers"][0]["args"][1] = boot
    spec["timeoutSeconds"] = "400"
    (data / "cloud-run-fresh-recovery.json").write_text(json.dumps(job, indent=2) + "\n")
    report = {
        "original_successful_records": len(queries),
        "failed_query": failed,
        "previous_reserved_attempts": old_attempts,
        "remaining_unattempted_tasks": len(pending),
        "maximum_cumulative_attempts": old_attempts + len(pending),
        "worker_timeout_seconds": 400,
        "previous_worker_elapsed_seconds_conservative": 792.145,
        "maximum_cumulative_worker_seconds": 1192.145,
        "same_job_definition": True,
        "same_inference_modules": True,
        "failed_query_retried": False,
        "policy_changed": False,
        "authorization_usd": 8,
        "reason": "Recover unattempted tasks after provider 504; retain original failure and uncertainty. No new spending envelope.",
    }
    Path("research/experiments/campaign-04/cloud-recovery.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
