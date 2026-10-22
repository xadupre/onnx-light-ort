"""Source-build frontend; release wheels use the pyproject.toml build backend."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


def positive_jobs(value):
    jobs = int(value)
    if jobs < 1:
        raise argparse.ArgumentTypeError("parallel jobs must be a positive integer")
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build_ext", help="build native libraries and model bindings")
    build.add_argument("--inplace", "-i", action="store_true", required=True)
    build.add_argument(
        "--cpp-tests", action="store_true", help="build and run native CTest tests"
    )
    build.add_argument(
        "--parallel",
        "-j",
        type=positive_jobs,
        default=os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count() or 1)),
    )
    build.add_argument("--dry-run", "-n", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    build_dir = root / "build"
    configure = [
        "cmake",
        "-S",
        str(root),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DSKBUILD=OFF",
        "-DONNX_LIGHT_ORT_BUILD_HOST=ON",
        "-DONNX_LIGHT_ORT_BUILD_PLUGIN=ON",
        "-DONNX_LIGHT_ORT_BUILD_PYTHON=ON",
        "-DONNX_LIGHT_ORT_BUILD_EXAMPLES=ON",
        "-DONNX_LIGHT_ORT_INSTALL=ON",
        "-DONNX_LIGHT_ORT_TEST_PYTHON=OFF",
        "-DONNX_LIGHT_ORT_TEST_WITH_ORT=OFF",
        f"-DONNX_LIGHT_ORT_BUILD_TESTS={'ON' if args.cpp_tests else 'OFF'}",
        *shlex.split(os.environ.get("CMAKE_ARGS", "")),
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]
    calls = [
        configure,
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--parallel",
            str(args.parallel),
        ],
    ]
    if args.cpp_tests:
        calls.append(
            [
                "ctest",
                "--test-dir",
                str(build_dir),
                "-C",
                "Release",
                "-L",
                "unit|dependency|consumer",
                "--output-on-failure",
                "--no-tests=error",
                "--parallel",
                str(args.parallel),
            ]
        )
    for command in calls:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
