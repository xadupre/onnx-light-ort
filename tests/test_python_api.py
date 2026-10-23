"""Executes the source-build defaults without loading ONNX Runtime."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy
from onnx_light.onnx import TensorProto, helper

from onnx_light_ort import CustomOpSessionOrtCalledByLight, get_library_path
from onnx_light_ort.libraries import _get_build_library_path

HOST = None
LIBRARY = None


class TestPythonApi(unittest.TestCase):
    def test_ctest_libraries_and_model_formats(self):
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Log", ["X"], ["Y"], domain="onnx_light_cpu")],
                "python-api",
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, None)],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
            ),
            opset_imports=[helper.make_opsetid("onnx_light_cpu", 1)],
            ir_version=9,
        )
        values = numpy.array([1, 2, 3], dtype=numpy.float32)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            path.write_bytes(model.SerializeToString())
            for source in (model, model.SerializeToString(), path, str(path)):
                with (
                    self.subTest(source=type(source)),
                    CustomOpSessionOrtCalledByLight(
                        source, library=LIBRARY, host_library=HOST
                    ) as session,
                ):
                    numpy.testing.assert_allclose(
                        session.run(None, {"X": values})[0], numpy.log(values), rtol=2e-6
                    )
            try:
                default_libraries = (
                    get_library_path("host"),
                    get_library_path("custom_ops"),
                )
            except FileNotFoundError:
                default_libraries = None
            if default_libraries == (HOST, LIBRARY):
                with CustomOpSessionOrtCalledByLight(model) as session:
                    numpy.testing.assert_allclose(
                        session.run(None, {"X": values})[0], numpy.log(values), rtol=2e-6
                    )
        self.assertNotIn("onnxruntime", sys.modules)
        self.assertNotIn("onnx", sys.modules)

    def test_missing_and_oversized_model_files(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            with self.assertRaises(FileNotFoundError):
                CustomOpSessionOrtCalledByLight(path, library=LIBRARY, host_library=HOST)
            with path.open("wb") as stream:
                stream.truncate(16 * 1024 * 1024 + 1)
            with self.assertRaisesRegex(ValueError, "16 MiB"):
                CustomOpSessionOrtCalledByLight(path, library=LIBRARY, host_library=HOST)

    def test_unknown_component(self):
        with self.assertRaisesRegex(ValueError, "Unknown native component"):
            get_library_path("../host")

    def test_manifest_uses_active_build_configuration(self):
        for cache, config in (
            ("CMAKE_BUILD_TYPE:STRING=\n", ""),
            ("CMAKE_BUILD_TYPE:STRING=Debug\n", "Debug"),
            ("CMAKE_BUILD_TYPE:STRING=Release\n", "Release"),
            ("CMAKE_CONFIGURATION_TYPES:STRING=Debug;Release;RelWithDebInfo\n", "Release"),
        ):
            with self.subTest(cache=cache), TemporaryDirectory() as directory:
                build = Path(directory)
                (build / "CMakeCache.txt").write_text(cache, encoding="utf-8")
                for name in ("", "Debug", "Release"):
                    destination = build / name
                    destination.mkdir(exist_ok=True)
                    library = destination / "native-library"
                    library.touch()
                    (destination / "python_libraries.json").write_text(
                        json.dumps({"host": str(library)}), encoding="utf-8"
                    )
                self.assertEqual(
                    _get_build_library_path(build, "host"),
                    build / config / "native-library",
                )
                with self.assertRaisesRegex(FileNotFoundError, "disabled"):
                    _get_build_library_path(build, "custom_ops")
                (build / config / "python_libraries.json").unlink()
                with self.assertRaises(FileNotFoundError):
                    _get_build_library_path(build, "host")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    HOST = arguments.host.resolve(strict=True)
    LIBRARY = arguments.library.resolve(strict=True)
    unittest.main(argv=[__file__, *remaining], verbosity=2)
