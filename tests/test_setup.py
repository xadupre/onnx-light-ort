"""Exercises the source-build frontend without another compiled tree."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


class TestSetup(unittest.TestCase):
    def run_setup(self, *args, environment=None):
        root = Path(__file__).resolve().parents[1]
        main = runpy.run_path(str(root / "setup.py"))["main"]
        calls = []

        def run(command, *, cwd, check):
            self.assertEqual(cwd, root)
            self.assertTrue(check)
            calls.append(command)

        main.__globals__["subprocess"] = SimpleNamespace(run=run)
        main.__globals__["os"] = SimpleNamespace(
            environ=os.environ if environment is None else environment,
            cpu_count=os.cpu_count,
        )
        stdout, stderr = StringIO(), StringIO()
        returncode = 0
        argv = sys.argv
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                sys.argv = [str(root / "setup.py"), "build_ext", "--inplace", *args]
                main()
            except SystemExit as error:
                returncode = error.code
            finally:
                sys.argv = argv
        if returncode == 0:
            self.assertEqual(stdout.getvalue().splitlines(), [shlex.join(c) for c in calls])
        return subprocess.CompletedProcess(
            calls, returncode, stdout=stdout.getvalue(), stderr=stderr.getvalue()
        )

    def test_removed_dry_run_options(self):
        for option in ("--dry-run", "-n"):
            with self.subTest(option=option):
                result = self.run_setup(option)
                self.assertEqual(result.returncode, 2)
                self.assertIn(f"unrecognized arguments: {option}", result.stderr)
                self.assertEqual(result.args, [])
                self.assertEqual(result.stdout, "")

    def test_cpp_tests_and_cmake_args(self):
        environment = os.environ.copy()
        environment["CMAKE_ARGS"] = '-DFETCHCONTENT_SOURCE_DIR_ONNX_LIGHT="/source with spaces"'
        result = self.run_setup("--cpp-tests", "--parallel", "2", environment=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.args), 3)
        self.assertIn("-DONNX_LIGHT_ORT_BUILD_TESTS=ON", result.stdout)
        self.assertIn("-DFETCHCONTENT_SOURCE_DIR_ONNX_LIGHT=/source with spaces", result.stdout)
        self.assertIn("-DPython3_EXECUTABLE=", result.stdout)
        self.assertIn("ctest", result.stdout)
        self.assertIn("--parallel 2", result.stdout)
        self.assertNotIn("--install", result.stdout)

    def test_default_does_not_run_tests(self):
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.args), 2)
        self.assertIn("-DONNX_LIGHT_ORT_BUILD_TESTS=OFF", result.stdout)
        self.assertNotIn("ctest", result.stdout)
        self.assertIn("-DONNX_LIGHT_ORT_ONNX_LIGHT_SOURCE_DIR=", result.stdout)

    def test_release_configuration_overrides_cmake_args(self):
        environment = os.environ.copy()
        environment["CMAKE_ARGS"] = "-DCMAKE_BUILD_TYPE=Debug"
        result = self.run_setup(environment=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        configure = shlex.split(result.stdout.splitlines()[0])
        defines = dict(arg[2:].split("=", 1) for arg in configure if arg.startswith("-D"))
        self.assertEqual(defines["CMAKE_BUILD_TYPE"], "Release")
        self.assertIn("--config Release", result.stdout)

    def test_local_onnx_light_runtime(self):
        import onnx_light

        info = onnx_light.get_cpp_build_info()
        environment = os.environ.copy()
        environment["CMAKE_ARGS"] = "-DONNX_LIGHT_ORT_ONNX_LIGHT_CORE_LIBRARY=/wrong-runtime"
        result = self.run_setup("--onnx-light-source", environment=environment)
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
            with self.assertRaisesRegex(FileNotFoundError, "Missing onnx-light source header:"):
                self.run_setup("--onnx-light-source", environment=environment)

    def test_invalid_parallelism(self):
        for value in ("0", "-1", "invalid"):
            with self.subTest(value=value):
                result = self.run_setup("--parallel", value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("error:", result.stderr)

    def test_parallelism_from_environment(self):
        environment = os.environ.copy()
        environment["CMAKE_BUILD_PARALLEL_LEVEL"] = "3"
        result = self.run_setup(environment=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--parallel 3", result.stdout)
        for value in ("0", "-1", "invalid"):
            with self.subTest(value=value):
                environment["CMAKE_BUILD_PARALLEL_LEVEL"] = value
                result = self.run_setup(environment=environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("error:", result.stderr)
        result = self.run_setup("--parallel", "2", environment=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--parallel 2", result.stdout)


if __name__ == "__main__":
    unittest.main()
