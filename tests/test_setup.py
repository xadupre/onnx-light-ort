"""Exercises the source-build frontend without another compiled tree."""

import os
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class TestSetup(unittest.TestCase):
    def run_setup(self, *args, environment=None):
        return subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "setup.py"),
                "build_ext",
                "--inplace",
                *args,
            ],
            env=environment,
            text=True,
            capture_output=True,
        )

    def test_cpp_tests_and_cmake_args(self):
        environment = os.environ.copy()
        environment["CMAKE_ARGS"] = '-DFETCHCONTENT_SOURCE_DIR_ONNX_LIGHT="/source with spaces"'
        result = self.run_setup(
            "--cpp-tests", "--dry-run", "--parallel", "2", environment=environment
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-DONNX_LIGHT_ORT_BUILD_TESTS=ON", result.stdout)
        self.assertIn("-DFETCHCONTENT_SOURCE_DIR_ONNX_LIGHT=/source with spaces", result.stdout)
        self.assertIn("-DPython3_EXECUTABLE=", result.stdout)
        self.assertIn("ctest", result.stdout)
        self.assertIn("--parallel 2", result.stdout)
        self.assertNotIn("--install", result.stdout)

    def test_default_does_not_run_tests(self):
        result = self.run_setup("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-DONNX_LIGHT_ORT_BUILD_TESTS=OFF", result.stdout)
        self.assertNotIn("ctest", result.stdout)
        self.assertIn("-DONNX_LIGHT_ORT_ONNX_LIGHT_SOURCE_DIR=", result.stdout)

    def test_local_onnx_light_runtime(self):
        import onnx_light

        info = onnx_light.get_cpp_build_info()
        environment = os.environ.copy()
        environment["CMAKE_ARGS"] = "-DONNX_LIGHT_ORT_ONNX_LIGHT_CORE_LIBRARY=/wrong-runtime"
        result = self.run_setup("--onnx-light-source", "--dry-run", environment=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        configure = shlex.split(result.stdout.splitlines()[0])
        defines = dict(arg[2:].split("=", 1) for arg in configure if arg.startswith("-D"))
        self.assertEqual(defines["ONNX_LIGHT_ORT_INSTALL"], "OFF")
        for component in ("core", "proto", "threads"):
            self.assertEqual(
                defines[f"ONNX_LIGHT_ORT_ONNX_LIGHT_{component.upper()}_LIBRARY"],
                str(Path(info[f"{component}_library"]).resolve(strict=True)),
            )
            if sys.platform == "win32":
                name = f"ONNX_LIGHT_ORT_ONNX_LIGHT_{component.upper()}_IMPLIB"
                expected = environment.get(name, info.get(f"{component}_import_library"))
                self.assertEqual(defines[name], str(Path(expected).resolve(strict=True)))
        self.assertEqual(
            defines["ONNX_LIGHT_ORT_ONNX_LIGHT_PYTHON_DIR"],
            str(Path(onnx_light.__file__).resolve().parent.parent),
        )

    def test_invalid_local_source(self):
        with TemporaryDirectory(prefix="onnx light source ") as directory:
            environment = os.environ.copy()
            environment["ONNX_LIGHT_ORT_ONNX_LIGHT_SOURCE_DIR"] = directory
            result = self.run_setup("--onnx-light-source", "--dry-run", environment=environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing onnx-light source header:", result.stderr)
            self.assertNotIn("cmake", result.stdout)

    def test_invalid_parallelism(self):
        for value in ("0", "-1", "invalid"):
            with self.subTest(value=value):
                result = self.run_setup("--dry-run", "--parallel", value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("error:", result.stderr)

    def test_parallelism_from_environment(self):
        environment = os.environ.copy()
        environment["CMAKE_BUILD_PARALLEL_LEVEL"] = "3"
        result = self.run_setup("--dry-run", environment=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--parallel 3", result.stdout)
        for value in ("0", "-1", "invalid"):
            with self.subTest(value=value):
                environment["CMAKE_BUILD_PARALLEL_LEVEL"] = value
                result = self.run_setup("--dry-run", environment=environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("error:", result.stderr)
        result = self.run_setup("--dry-run", "--parallel", "2", environment=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--parallel 2", result.stdout)


if __name__ == "__main__":
    unittest.main()
