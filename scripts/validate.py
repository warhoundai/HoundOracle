"""Reproduce local acceptance checks and preserve actual command outcomes."""

import json
import subprocess
import time
from pathlib import Path

checks = [
    ["uv", "run", "ruff", "check", "src", "tests", "examples", "benchmarks", "scripts"],
    ["uv", "run", "ruff", "format", "--check", "src", "tests", "examples", "benchmarks", "scripts"],
    ["uv", "run", "pytest", "-q"],
    ["uv", "run", "python", "-m", "build"],
    ["uv", "run", "pip-audit", "--progress-spinner", "off"],
    ["uv", "run", "python", "benchmarks/run.py", "--output", "benchmarks/local-result.json"],
]


def main():
    results = []
    for command in checks:
        start = time.monotonic()
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        results.append(
            {
                "command": command,
                "exit_code": result.returncode,
                "seconds": time.monotonic() - start,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        print("PASS" if result.returncode == 0 else "FAIL", " ".join(command), flush=True)
    Path("docs/validation.json").write_text(
        json.dumps({"schema_version": 1, "checks": results}, indent=2) + "\n"
    )
    raise SystemExit(0 if all(r["exit_code"] == 0 for r in results) else 1)


if __name__ == "__main__":
    main()
