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


def local_onnx_light_defines():
    """Selects the headers and shared libraries of the imported onnx-light runtime."""
    import onnx_light

    info = onnx_light.get_cpp_build_info()
    source = Path(
        os.environ.get(
            "ONNX_LIGHT_ORT_ONNX_LIGHT_SOURCE_DIR", str(Path(info["include_dir"]).parent)
        )
    ).resolve()
    for header in (
        "onnx_light/onnx_core/runtime/kernels/kernel_dispatch_table.h",
        "onnx_proto/include/onnx_proto/onnx.h",
        "onnx_threads/include/onnx_threads/config.h",
    ):
        if not (source / header).is_file():
            raise FileNotFoundError(
                f"Missing onnx-light source header: {source / header}. Set "
                "ONNX_LIGHT_ORT_ONNX_LIGHT_SOURCE_DIR to the matching source checkout."
            )
    defines = [
        f"-DONNX_LIGHT_ORT_ONNX_LIGHT_SOURCE_DIR={source}",
        f"-DONNX_LIGHT_ORT_ONNX_LIGHT_PYTHON_DIR={Path(onnx_light.__file__).resolve().parent.parent}",
    ]
    for component in ("core", "proto", "threads"):
        library = Path(info[f"{component}_library"]).resolve(strict=True)
        prefix = f"ONNX_LIGHT_ORT_ONNX_LIGHT_{component.upper()}"
        defines.append(f"-D{prefix}_LIBRARY={library}")
        if sys.platform == "win32":
            import_path = os.environ.get(
                f"{prefix}_IMPLIB", info.get(f"{component}_import_library")
            )
            if not import_path:
                raise FileNotFoundError(
                    f"Missing matching Windows import library; set {prefix}_IMPLIB."
                )
            import_library = Path(import_path).resolve(strict=True)
            defines.append(f"-D{prefix}_IMPLIB={import_library}")
    return defines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build_ext", help="build native libraries and model bindings")
    build.add_argument("--inplace", "-i", action="store_true", required=True)
    build.add_argument(
        "--cpp-tests", action="store_true", help="build and run native CTest tests"
    )
    build.add_argument(
        "--onnx-light-source",
        action="store_true",
        help="use the local onnx-light headers and already-built Python runtime",
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
        "-DSKBUILD=OFF",
        "-DONNX_LIGHT_ORT_BUILD_HOST=ON",
        "-DONNX_LIGHT_ORT_BUILD_PLUGIN=ON",
        "-DONNX_LIGHT_ORT_BUILD_PYTHON=ON",
        "-DONNX_LIGHT_ORT_BUILD_EXAMPLES=ON",
        f"-DONNX_LIGHT_ORT_INSTALL={'OFF' if args.onnx_light_source else 'ON'}",
        "-DONNX_LIGHT_ORT_ONNX_LIGHT_SOURCE_DIR=",
        "-DONNX_LIGHT_ORT_TEST_PYTHON=OFF",
        "-DONNX_LIGHT_ORT_TEST_WITH_ORT=OFF",
        f"-DONNX_LIGHT_ORT_BUILD_TESTS={'ON' if args.cpp_tests else 'OFF'}",
        *shlex.split(os.environ.get("CMAKE_ARGS", "")),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]
    if args.onnx_light_source:
        configure.extend(local_onnx_light_defines())
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
