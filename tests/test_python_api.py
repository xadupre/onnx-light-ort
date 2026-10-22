"""Executes the source-build defaults without loading ONNX Runtime."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy
from onnx_light.onnx import TensorProto, helper

from onnx_light_ort import CustomOpSessionOrtCalledByLight, get_library_path


class TestPythonApi(unittest.TestCase):
    def test_default_libraries_and_model_formats(self):
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
                    CustomOpSessionOrtCalledByLight(source) as session,
                ):
                    numpy.testing.assert_allclose(
                        session.run(None, {"X": values})[0], numpy.log(values), rtol=2e-6
                    )
        self.assertNotIn("onnxruntime", sys.modules)
        self.assertNotIn("onnx", sys.modules)

    def test_missing_and_oversized_model_files(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            with self.assertRaises(FileNotFoundError):
                CustomOpSessionOrtCalledByLight(path)
            with path.open("wb") as stream:
                stream.truncate(16 * 1024 * 1024 + 1)
            with self.assertRaisesRegex(ValueError, "16 MiB"):
                CustomOpSessionOrtCalledByLight(path)

    def test_unknown_component(self):
        with self.assertRaisesRegex(ValueError, "Unknown native component"):
            get_library_path("../host")


if __name__ == "__main__":
    unittest.main()
