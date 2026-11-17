"""Own one RAM checkpoint directory and clean it after success, failure or timeout."""

import json
import os
import subprocess
from pathlib import Path


def main():
    directory = Path("/dev/shm/houndoracle-reasoning40")
    directory.mkdir()  # Refuse to assume ownership of an existing directory.
    private = Path("/workspace/houndoracle-data/campaign-40")
    code = 1
    try:
        with (private / "run.log").open("x") as log:
            try:
                result = subprocess.run(
                    [
                        "/workspace/houndoracle-reasoning/bin/python",
                        "scripts/local_reasoning_screen.py",
                    ],
                    env=os.environ | {"PYTHONPATH": "src"},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=2400,
                    check=False,
                )
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
    finally:
        model = directory / "model.gguf"
        if model.is_file() or model.is_symlink():
            model.unlink()
        directory.rmdir()
        (private / "exit.json").write_text(
            json.dumps({"returncode": code, "owned_ram_directory_removed": not directory.exists()})
            + "\n"
        )
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
