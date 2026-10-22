import argparse
import gc
from pathlib import Path
import sys
import unittest

import numpy
import onnxruntime
from onnx_light.onnx import ModelProto, TensorProto, helper
from onnxruntime.capi.onnxruntime_pybind11_state import (
    Fail,
    InvalidArgument,
    InvalidGraph,
    RuntimeException,
)

from cpu_custom_ops_test_cases import CpuCustomOpCases, TypedCustomOpCases, make_cpu_log_model

LIBRARY = None


def make_model(alpha=0.125, dtype=TensorProto.FLOAT):
    """Creates a model with an explicit onnx-light custom operator."""
    attributes = {} if alpha is None else {"alpha": alpha}
    graph = helper.make_graph(
        [helper.make_node("LeakyRelu", ["X"], ["Y"], domain="onnx_light", **attributes)],
        "onnx-light-custom-op",
        [helper.make_tensor_value_info("X", dtype, None)],
        [helper.make_tensor_value_info("Y", dtype, None)],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("onnx_light", 1),
        ],
        ir_version=9,
    ).SerializeToString()


def make_session(model=None):
    """Creates a real ORT session loading the compiled plugin."""
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.register_custom_ops_library(str(LIBRARY))
    return onnxruntime.InferenceSession(
        make_model() if model is None else model,
        options,
        providers=["CPUExecutionProvider"],
    )


class TestCustomOpsLightCalledByOrt(unittest.TestCase):
    def test_changing_shapes_and_readonly_inputs(self):
        session = make_session()
        for shape in ((4,), (2, 3), (), (0,), (2, 0, 3), (1,) * 16):
            with self.subTest(shape=shape):
                count = int(numpy.prod(shape)) if shape else 1
                values = (numpy.arange(count, dtype=numpy.float32) - 2).reshape(shape)
                original = values.copy()
                values.setflags(write=False)
                for _ in range(3):
                    result = session.run(None, {"X": values})[0]
                    expected = numpy.where(values < 0, values * numpy.float32(0.125), values)
                    numpy.testing.assert_array_equal(result, expected)
                    numpy.testing.assert_array_equal(values, original)
                    self.assertEqual(result.dtype, numpy.float32)
                    self.assertEqual(result.shape, shape)
        self.assertNotIn("onnx", sys.modules)

    def test_preallocated_output(self):
        session = make_session()
        values = numpy.array([-4, -1, 0, 2], dtype=numpy.float32)
        output = numpy.full(values.shape, numpy.nan, dtype=numpy.float32)
        bound_input = onnxruntime.OrtValue.ortvalue_from_numpy(values)
        bound_output = onnxruntime.OrtValue.ortvalue_from_numpy(output)
        binding = session.io_binding()
        binding.bind_ortvalue_input("X", bound_input)
        binding.bind_ortvalue_output("Y", bound_output)
        session.run_with_iobinding(binding)
        numpy.testing.assert_array_equal(output, numpy.array([-0.5, -0.125, 0, 2], numpy.float32))
        self.assertEqual(binding.get_outputs()[0].data_ptr(), bound_output.data_ptr())

    def test_attributes_and_sessions_are_independent(self):
        first = make_session(make_model(alpha=0.25))
        second = make_session(make_model(alpha=0.5))
        values = numpy.array([-4, 2], dtype=numpy.float32)
        numpy.testing.assert_array_equal(first.run(None, {"X": values})[0], [-1, 2])
        numpy.testing.assert_array_equal(second.run(None, {"X": values})[0], [-2, 2])
        del first
        gc.collect()
        numpy.testing.assert_array_equal(second.run(None, {"X": values})[0], [-2, 2])

    def test_invalid_attributes(self):
        for alpha in (None, "not-a-float"):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(RuntimeException, "attribute|Attribute"):
                    make_session(make_model(alpha=alpha))

    def test_wrong_arity(self):
        for inputs in ([], ["X", "X"]):
            with self.subTest(inputs=inputs):
                model = ModelProto()
                model.ParseFromString(make_model())
                del model.graph.node[0].input[:]
                model.graph.node[0].input.extend(inputs)
                with self.assertRaisesRegex(InvalidGraph, "input|Input"):
                    make_session(model.SerializeToString())

    def test_unaligned_buffers(self):
        for element_type, dtype, offset in (
            (TensorProto.FLOAT16, numpy.float16, 1),
            (TensorProto.FLOAT, numpy.float32, 1),
            (TensorProto.DOUBLE, numpy.float64, 4),
        ):
            session = make_session(make_model(dtype=element_type))
            for side in ("input", "output"):
                with self.subTest(dtype=dtype, side=side):
                    unaligned = numpy.ndarray(
                        (4,),
                        dtype=dtype,
                        buffer=bytearray(4 * numpy.dtype(dtype).itemsize + offset),
                        offset=offset,
                    )
                    values = unaligned if side == "input" else numpy.empty(4, dtype)
                    output = unaligned if side == "output" else numpy.empty(4, dtype)
                    values[:] = [-4, -1, 0, 2]
                    binding = session.io_binding()
                    binding.bind_input("X", "cpu", 0, dtype, values.shape, values.ctypes.data)
                    binding.bind_output("Y", "cpu", 0, dtype, output.shape, output.ctypes.data)
                    with self.assertRaisesRegex(RuntimeError, "element-aligned buffers"):
                        session.run_with_iobinding(binding)

    def test_overlapping_buffers(self):
        for element_type, dtype in (
            (TensorProto.FLOAT16, numpy.float16),
            (TensorProto.FLOAT, numpy.float32),
            (TensorProto.DOUBLE, numpy.float64),
        ):
            session = make_session(make_model(dtype=element_type))
            for offset in (0, 1, -1, 3, -2):
                with self.subTest(dtype=dtype, offset=offset):
                    storage = numpy.arange(10, dtype=dtype) - 2
                    original = storage.copy()
                    values = storage[2:6]
                    output = storage[2 + offset : 6 + offset]
                    binding = session.io_binding()
                    binding.bind_input("X", "cpu", 0, dtype, values.shape, values.ctypes.data)
                    binding.bind_output("Y", "cpu", 0, dtype, output.shape, output.ctypes.data)
                    with self.assertRaisesRegex(RuntimeError, "non-overlapping buffers"):
                        session.run_with_iobinding(binding)
                    numpy.testing.assert_array_equal(storage, original)

    def test_wrong_type_and_unsupported_rank(self):
        with self.assertRaisesRegex((Fail, InvalidArgument, InvalidGraph), "[Tt]ype|float"):
            make_session(make_model(dtype=TensorProto.INT32))
        session = make_session()
        with self.assertRaisesRegex(InvalidArgument, "16 dimensions"):
            session.run(None, {"X": numpy.ones((1,) * 17, dtype=numpy.float32)})


class TestCpuCustomOpsLightCalledByOrt(CpuCustomOpCases, TypedCustomOpCases, unittest.TestCase):
    input_errors = InvalidArgument

    def make_cpu_session(self):
        return make_session(make_cpu_log_model())

    def make_typed_session(self, model):
        return make_session(model)

    def run_typed(self, session, values, element_type):
        storage = values.view(numpy.uint16) if element_type == TensorProto.BFLOAT16 else values
        output = numpy.empty_like(values)
        output_storage = (
            output.view(numpy.uint16) if element_type == TensorProto.BFLOAT16 else output
        )
        binding = session.io_binding()
        binding.bind_ortvalue_input(
            "X", onnxruntime.OrtValue.ortvalue_from_numpy_with_onnx_type(storage, element_type)
        )
        bound_output = onnxruntime.OrtValue.ortvalue_from_numpy_with_onnx_type(
            output_storage, element_type
        )
        binding.bind_ortvalue_output("Y", bound_output)
        session.run_with_iobinding(binding)
        self.assertEqual(binding.get_outputs()[0].data_ptr(), bound_output.data_ptr())
        return output

    def test_cpu_log_preallocated_output(self):
        session = self.make_cpu_session()
        values = numpy.linspace(0.01, 10, 33, dtype=numpy.float32)
        output = numpy.empty_like(values)
        binding = session.io_binding()
        binding.bind_ortvalue_input("X", onnxruntime.OrtValue.ortvalue_from_numpy(values))
        bound_output = onnxruntime.OrtValue.ortvalue_from_numpy(output)
        binding.bind_ortvalue_output("Y", bound_output)
        session.run_with_iobinding(binding)
        numpy.testing.assert_allclose(output, numpy.log(values), rtol=2e-6, atol=0)
        self.assertEqual(binding.get_outputs()[0].data_ptr(), bound_output.data_ptr())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    LIBRARY = arguments.library.resolve(strict=True)
    unittest.main(argv=[__file__, *remaining], verbosity=2)
