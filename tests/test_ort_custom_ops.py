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


class TestCustomOps(unittest.TestCase):
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
        session = make_session()
        for side in ("input", "output"):
            with self.subTest(side=side):
                unaligned = numpy.ndarray(
                    (4,), dtype=numpy.float32, buffer=bytearray(17), offset=1
                )
                values = unaligned if side == "input" else numpy.empty(4, numpy.float32)
                output = unaligned if side == "output" else numpy.empty(4, numpy.float32)
                values[:] = [-4, -1, 0, 2]
                binding = session.io_binding()
                binding.bind_input("X", "cpu", 0, numpy.float32, values.shape, values.ctypes.data)
                binding.bind_output(
                    "Y", "cpu", 0, numpy.float32, output.shape, output.ctypes.data
                )
                with self.assertRaisesRegex(RuntimeError, "float-aligned buffers"):
                    session.run_with_iobinding(binding)

    def test_overlapping_buffers(self):
        session = make_session()
        for offset in (0, 1, -1):
            with self.subTest(offset=offset):
                storage = numpy.arange(6, dtype=numpy.float32) - 2
                original = storage.copy()
                values = storage[1:5]
                output = storage[1 + offset : 5 + offset]
                binding = session.io_binding()
                binding.bind_input("X", "cpu", 0, numpy.float32, values.shape, values.ctypes.data)
                binding.bind_output(
                    "Y", "cpu", 0, numpy.float32, output.shape, output.ctypes.data
                )
                with self.assertRaisesRegex(RuntimeError, "non-overlapping buffers"):
                    session.run_with_iobinding(binding)
                numpy.testing.assert_array_equal(storage, original)

    def test_wrong_type_and_unsupported_rank(self):
        with self.assertRaisesRegex((Fail, InvalidArgument, InvalidGraph), "[Tt]ype|float"):
            make_session(make_model(dtype=TensorProto.DOUBLE))
        session = make_session()
        with self.assertRaisesRegex(InvalidArgument, "16 dimensions"):
            session.run(None, {"X": numpy.ones((1,) * 17, dtype=numpy.float32)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    LIBRARY = arguments.library.resolve(strict=True)
    unittest.main(argv=[__file__, *remaining], verbosity=2)
