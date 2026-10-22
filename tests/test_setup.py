"""Exercises the source-build frontend without another compiled tree."""

import os
from pathlib import Path
import subprocess
import sys
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
