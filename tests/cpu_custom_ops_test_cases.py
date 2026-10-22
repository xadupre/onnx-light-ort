"""Numerical cases shared by the two real hosts of the CPU custom operator."""

import sys

import numpy
from ml_dtypes import bfloat16
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


class TypedCustomOpCases:
    def test_supported_native_types(self):
        types = (
            (TensorProto.FLOAT, numpy.float32, 2e-6),
            (TensorProto.DOUBLE, numpy.float64, 2e-13),
            (TensorProto.FLOAT16, numpy.float16, 2e-3),
            (TensorProto.BFLOAT16, bfloat16, 2e-2),
        )
        for op, domain in (("LeakyRelu", "onnx_light"), ("Log", "onnx_light_cpu")):
            for element_type, dtype, tolerance in types:
                attributes = {"alpha": 0.125} if op == "LeakyRelu" else {}
                model = helper.make_model(
                    helper.make_graph(
                        [helper.make_node(op, ["X"], ["Y"], domain=domain, **attributes)],
                        "typed-custom-op",
                        [helper.make_tensor_value_info("X", element_type, None)],
                        [helper.make_tensor_value_info("Y", element_type, None)],
                    ),
                    opset_imports=[helper.make_opsetid(domain, 1)],
                    ir_version=9,
                )
                session = self.make_typed_session(model.SerializeToString())
                arrays = [
                    numpy.linspace(-2 if op == "LeakyRelu" else 0.125, 10, count)
                    .astype(dtype)
                    .reshape(shape)
                    for shape, count in (
                        ((), 1),
                        ((0,), 0),
                        ((2, 0, 3), 0),
                        ((3, 7), 21),
                        ((257,), 257),
                    )
                ]
                arrays.append(
                    numpy.array([-numpy.inf, -1, -0.0, 0, 1, numpy.inf, numpy.nan], dtype=dtype)
                )
                if dtype == numpy.float64:
                    arrays.append(numpy.array([1 - 2**-40, 1 + 2**-40, 1e100], dtype=dtype))
                for values in arrays:
                    with self.subTest(op=op, dtype=dtype, shape=values.shape):
                        original = values.copy()
                        values.setflags(write=False)
                        wide = values.astype(numpy.float64)
                        with numpy.errstate(divide="ignore", invalid="ignore"):
                            expected = (
                                numpy.where(wide < 0, wide * 0.125, wide)
                                if op == "LeakyRelu"
                                else numpy.log(wide)
                            ).astype(dtype)
                        previous = None
                        for _ in range(2):
                            result = self.run_typed(session, values, element_type)
                            self.assertEqual(result.dtype, numpy.dtype(dtype))
                            self.assertEqual(result.shape, values.shape)
                            numpy.testing.assert_allclose(
                                result.astype(numpy.float64),
                                expected.astype(numpy.float64),
                                rtol=tolerance,
                                atol=0,
                                equal_nan=True,
                            )
                            if previous is not None:
                                self.assertFalse(numpy.shares_memory(previous, result))
                            previous = result
                        numpy.testing.assert_array_equal(
                            values.astype(numpy.float64), original.astype(numpy.float64)
                        )
                        if op == "LeakyRelu":
                            numpy.testing.assert_array_equal(
                                numpy.signbit(result[values == 0]),
                                numpy.signbit(values[values == 0]),
                            )
