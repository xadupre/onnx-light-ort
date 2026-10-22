"""Numerical cases shared by the two real hosts of the CPU custom operator."""

import sys

import numpy
from onnx_light.onnx import TensorProto, helper


def make_cpu_log_model():
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("Log", ["X"], ["Y"], domain="onnx_light_cpu")],
            "cpu-log-custom-op",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, None)],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        ),
        opset_imports=[helper.make_opsetid("onnx_light_cpu", 1)],
        ir_version=9,
    ).SerializeToString()


class CpuCustomOpCases:
    def check_log(self, session, values):
        original = values.copy()
        values.setflags(write=False)
        with numpy.errstate(divide="ignore", invalid="ignore"):
            expected = numpy.log(values.astype(numpy.float64)).astype(numpy.float32)
        for _ in range(2):
            result = session.run(None, {"X": values})[0]
            numpy.testing.assert_allclose(result, expected, rtol=2e-6, atol=0, equal_nan=True)
            numpy.testing.assert_array_equal(values, original)
            self.assertEqual(result.dtype, numpy.float32)
            self.assertEqual(result.shape, values.shape)
        self.assertNotIn("onnx", sys.modules)

    def test_cpu_log_shapes_and_simd_tails(self):
        session = self.make_cpu_session()
        shapes = [(), (0,), (2, 0, 3), (3, 7), (1,) * 16]
        shapes.extend((size,) for size in (1, 7, 8, 15, 16, 17, 31, 32, 33, 255, 256, 257, 65536))
        for shape in shapes:
            with self.subTest(shape=shape):
                count = int(numpy.prod(shape)) if shape else 1
                self.check_log(
                    session, numpy.linspace(0.01, 10, count, dtype=numpy.float32).reshape(shape)
                )

    def test_cpu_log_special_values(self):
        session = self.make_cpu_session()
        values = numpy.array(
            [
                -numpy.inf,
                -numpy.finfo(numpy.float32).tiny,
                -1,
                -0.0,
                0,
                1,
                numpy.nextafter(numpy.float32(0), numpy.float32(1)),
                numpy.finfo(numpy.float32).tiny,
                1e-20,
                0.99999994,
                1.0000001,
                numpy.finfo(numpy.float32).max,
                numpy.nan,
                numpy.inf,
                -numpy.inf,
            ],
            dtype=numpy.float32,
        )
        self.check_log(session, numpy.tile(values, 5))

    def test_cpu_log_rejects_invalid_inputs_and_recovers(self):
        session = self.make_cpu_session()
        with self.assertRaisesRegex(self.input_errors, "[Tt]ype|float"):
            session.run(None, {"X": numpy.ones(3, dtype=numpy.float64)})
        with self.assertRaisesRegex(self.input_errors, "16"):
            session.run(None, {"X": numpy.ones((1,) * 17, dtype=numpy.float32)})
        self.check_log(session, numpy.array([0.125, 1, 10], dtype=numpy.float32))
