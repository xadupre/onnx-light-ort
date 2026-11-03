"""Reads native dependency versions from pyproject.toml for CI and releases."""

import argparse
from pathlib import Path
import tomllib


def dependency_versions(root):
    with (root / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["tool"]["onnx-light-ort"]["dependencies"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dependency", nargs="?", choices=("onnx-light", "onnx-light-cpu"))
    args = parser.parse_args()
    versions = dependency_versions(Path(__file__).resolve().parents[2])
    if args.dependency:
        print(versions[args.dependency])
    else:
        for name, version in versions.items():
            print(f"{name.replace('-', '_')}={version}")
