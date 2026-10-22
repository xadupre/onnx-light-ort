"""Test the installed wheel using the model bindings built from its own sources."""

import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
environment = os.environ.copy()
paths = [str(root / "build" / "python")]
if environment.get("PYTHONPATH"):
    paths.append(environment["PYTHONPATH"])
environment["PYTHONPATH"] = os.pathsep.join(paths)
if sys.platform == "win32":
    environment["PATH"] = (
        str(root / "build" / "python" / "onnx_light" / "onnx_py")
        + os.pathsep
        + environment["PATH"]
    )
subprocess.run(
    [
        sys.executable,
        str(root / "cmake" / "run_python_with_dll_paths.py"),
        str(root / ".github" / "scripts" / "check_release_package.py"),
    ],
    env=environment,
    check=True,
)
