"""Reject tags inconsistent with the CMake, Python and package versions."""

import argparse
from pathlib import Path
import re
import tomllib


def check_version(root, tag):
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = metadata["project"]["version"]
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    init = (root / "onnx_light_ort" / "__init__.py").read_text(encoding="utf-8")
    cmake_version = re.search(r"project\(onnx_light_ort VERSION ([0-9.]+) ", cmake)
    python_version = re.search(r'^__version__ = "([^"]+)"$', init, re.MULTILINE)
    if cmake_version is None or python_version is None:
        raise ValueError("Cannot find the CMake or Python package version")
    versions = {version, cmake_version[1], python_version[1]}
    if len(versions) != 1:
        raise ValueError(f"Inconsistent project versions: {sorted(versions)}")
    if tag and tag.removeprefix("v") != version:
        raise ValueError(f"Tag {tag!r} does not match project version {version!r}")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", nargs="?", default="")
    args = parser.parse_args()
    print(check_version(Path(__file__).resolve().parents[2], args.tag))
