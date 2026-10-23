"""Validates release metadata and pins the sdist's onnx-light revision."""

import argparse
from pathlib import Path
import re
import tomllib


def pin_onnx_light_revision(source, revision):
    """Updates the unique cached onnx-light revision without changing CMake formatting."""
    pattern = (
        r"(^[ \t]*set\s*\(\s*ONNX_LIGHT_ORT_ONNX_LIGHT_REVISION\s+)"
        r'(?:"[^"]*"|[^\s)]+)(?=\s+CACHE\s+STRING\b)'
    )
    pinned, count = re.subn(
        pattern, lambda match: f'{match[1]}"{revision}"', source, flags=re.MULTILINE
    )
    if count != 1:
        raise ValueError(f"Expected one cached onnx-light revision assignment, found {count}")
    return pinned


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
