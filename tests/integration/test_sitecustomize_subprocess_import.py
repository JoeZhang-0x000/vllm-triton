from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys


def test_sitecustomize_applies_host_patch_in_subprocess() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(repo_root)
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "print(json.dumps({"
                "'sitecustomize': 'sitecustomize' in sys.modules, "
                "'cpuinfo': 'cpuinfo' in sys.modules, "
                "'vllm_xpu': 'vllm_xpu' in sys.modules"
                "}))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    payload = json.loads(completed.stdout.strip())
    assert payload == {
        "sitecustomize": True,
        "cpuinfo": True,
        "vllm_xpu": False,
    }
