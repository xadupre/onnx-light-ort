"""Release-only smoke tests of installed wheel payloads, never source-tree substitutes."""

import ctypes
import importlib.metadata
from pathlib import Path
import sys
import unittest

import numpy

import onnx_light_ort
from onnx_light_ort import CustomOpSession, get_library_path


class TestReleasePackage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = get_library_path("host")
        cls.plugin = get_library_path("custom_ops")
        cls.native_host = ctypes.CDLL(str(cls.host))
        cls.native_plugin = ctypes.CDLL(str(cls.plugin))

    def test_installed_libraries_and_metadata(self):
        self.assertNotIn("onnx_light", sys.modules)
        package = Path(onnx_light_ort.__file__).resolve().parent
        self.assertEqual(self.host.parent, package / "_native")
        self.assertEqual(self.plugin.parent, package / "_native")
        self.assertEqual(importlib.metadata.version("onnx-light-ort"), onnx_light_ort.__version__)
        self.assertTrue(hasattr(self.native_host, "LightOrtCreateSession"))
        self.assertTrue(hasattr(self.native_plugin, "RegisterCustomOps"))
        self.assertIn(
            "onnx-light-ort: " + onnx_light_ort.__version__,
            (package / "_native" / "release-build-info.txt").read_text(encoding="utf-8"),
        )
        for dependency in (
            "onnx-light",
            "onnx-light-ort",
            "ml_cpu_compute",
            "onnxruntime",
            "dlpack",
        ):
            self.assertTrue((package / "_native" / "licenses" / dependency / "LICENSE").is_file())
        with self.assertRaisesRegex(ValueError, "Unknown native component"):
            get_library_path("../host")

    def test_packaged_custom_ops_in_standalone_host(self):
        from onnx_light.onnx import TensorProto, helper

        for op_type, domain, attributes in (
            ("LeakyRelu", "onnx_light", {"alpha": 0.125}),
            ("Log", "onnx_light_cpu", {}),
        ):
            with self.subTest(op_type=op_type):
                model = helper.make_model(
                    helper.make_graph(
                        [helper.make_node(op_type, ["X"], ["Y"], domain=domain, **attributes)],
                        "release-custom-ops",
                        [helper.make_tensor_value_info("X", TensorProto.FLOAT, None)],
                        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
                    ),
                    opset_imports=[helper.make_opsetid(domain, 1)],
                    ir_version=9,
                )
                with CustomOpSession(
                    model, library=self.plugin, host_library=self.host
                ) as session:
                    for shape in ((33,), (3, 7), (), (0,)):
                        count = int(numpy.prod(shape)) if shape else 1
                        values = numpy.linspace(0.01, 10, count, dtype=numpy.float32).reshape(
                            shape
                        )
                        if op_type == "LeakyRelu":
                            values -= 2
                            expected = numpy.where(values < 0, values * 0.125, values)
                        else:
                            expected = numpy.log(values.astype(numpy.float64)).astype(
                                numpy.float32
                            )
                        result = session.run(None, {"X": values})[0]
                        numpy.testing.assert_allclose(result, expected, rtol=2e-6, atol=0)
                        self.assertEqual(result.shape, shape)
        self.assertNotIn("onnxruntime", sys.modules)
        self.assertNotIn("onnx", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
