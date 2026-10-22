"""Runs the actual CPU plugin through the standalone host, without ORT."""

import argparse
import ctypes
from pathlib import Path
import sys
import unittest

import numpy
from onnx_light.onnx import ModelProto, TensorProto
from onnx_light_ort import CustomOpSessionOrtCalledByLight

from cpu_custom_ops_test_cases import CpuCustomOpCases, TypedCustomOpCases, make_cpu_log_model

HOST = None
LIBRARY = None


class TestCpuCustomOpsOrtCalledByLight(CpuCustomOpCases, TypedCustomOpCases, unittest.TestCase):
    input_errors = (TypeError, ValueError)

    def make_cpu_session(self):
        session = CustomOpSessionOrtCalledByLight(
            make_cpu_log_model(), library=LIBRARY, host_library=HOST
        )
        self.addCleanup(session.close)
        return session

    def make_typed_session(self, model):
        session = CustomOpSessionOrtCalledByLight(model, library=LIBRARY, host_library=HOST)
        self.addCleanup(session.close)
        return session

    def run_typed(self, session, values, element_type):
        return session.run(None, {"X": values})[0]

    def test_declared_types_and_limits(self):
        model = ModelProto()
        model.ParseFromString(make_cpu_log_model())
        model.graph.input[0].type.tensor_type.elem_type = TensorProto.DOUBLE
        with self.assertRaisesRegex(RuntimeError, "types must match"):
            self.make_typed_session(model)
        model.graph.output[0].type.tensor_type.elem_type = TensorProto.DOUBLE
        session = self.make_typed_session(model)
        legacy = session._api.LightOrtRun
        legacy.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        legacy.restype = ctypes.c_void_p
        scalar = numpy.array(1, numpy.float32)
        output = ctypes.POINTER(ctypes.c_float)()
        dimensions = ctypes.POINTER(ctypes.c_int64)()
        rank, count = ctypes.c_size_t(), ctypes.c_size_t()
        with self.assertRaisesRegex(RuntimeError, "tensor type"):
            session._check(
                legacy(
                    session._handle,
                    scalar.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    1,
                    None,
                    0,
                    ctypes.byref(output),
                    ctypes.byref(dimensions),
                    ctypes.byref(rank),
                    ctypes.byref(count),
                )
            )
        with self.assertRaisesRegex(TypeError, "float64"):
            session.run(None, {"X": numpy.ones(1, numpy.float32)})
        oversized = numpy.broadcast_to(numpy.array(1, numpy.float64), (8 * 1024 * 1024 + 1,))
        with self.assertRaisesRegex(ValueError, "64 MiB"):
            session.run(None, {"X": oversized})
        values = numpy.array([1 + 2**-40, 2], numpy.float64)
        numpy.testing.assert_allclose(
            session.run(None, {"X": values})[0], numpy.log(values), rtol=2e-13, atol=0
        )

    def tearDown(self):
        self.assertNotIn("onnxruntime", sys.modules)
        self.assertNotIn("onnx", sys.modules)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    HOST = arguments.host.resolve(strict=True)
    LIBRARY = arguments.library.resolve(strict=True)
    unittest.main(argv=[__file__, *remaining], verbosity=2)
