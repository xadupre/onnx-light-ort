"""Checks real plugin execution in onnx-light without importing ONNX Runtime."""

import argparse
import ctypes
from pathlib import Path
import sys
import unittest

import numpy
from onnx_light.onnx import ModelProto, TensorProto, helper

from onnx_light_ort import CustomOpSession

HOST = None
LIBRARY = None
BAD_LIBRARY = None


def make_model(alpha=0.125, *, version=1, dtype=TensorProto.FLOAT):
    """Creates a single-node model for the independent ORT custom-op library."""
    attributes = {} if alpha is None else {"alpha": alpha}
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("LeakyRelu", ["X"], ["Y"], domain="ort_custom_ops", **attributes)],
            "ort-plugin-in-onnx-light",
            [helper.make_tensor_value_info("X", dtype, None)],
            [helper.make_tensor_value_info("Y", dtype, None)],
        ),
        opset_imports=[helper.make_opsetid("ort_custom_ops", version)],
        ir_version=9,
    )


def make_session(model=None, library=None):
    """Creates the real native host with an explicitly selected plugin."""
    return CustomOpSession(
        make_model() if model is None else model,
        library=LIBRARY if library is None else library,
        host_library=HOST,
    )


class TestHostedCustomOps(unittest.TestCase):
    def test_shapes_readonly_inputs_and_independent_outputs(self):
        with make_session() as session:
            for shape in ((4,), (2, 3), (), (0,), (2, 0, 3), (1,) * 16):
                with self.subTest(shape=shape):
                    count = int(numpy.prod(shape)) if shape else 1
                    values = (numpy.arange(count, dtype=numpy.float32) - 2).reshape(shape)
                    original = values.copy()
                    values.setflags(write=False)
                    previous = None
                    for _ in range(3):
                        result = session.run(None, {"X": values})[0]
                        expected = numpy.where(values < 0, values * numpy.float32(0.125), values)
                        numpy.testing.assert_array_equal(result, expected)
                        numpy.testing.assert_array_equal(values, original)
                        self.assertEqual(result.shape, shape)
                        self.assertEqual(result.dtype, numpy.float32)
                        if previous is not None:
                            numpy.testing.assert_array_equal(previous, result)
                            self.assertFalse(numpy.shares_memory(previous, result))
                        previous = result
            self.assertNotIn("onnxruntime", sys.modules)
            self.assertNotIn("onnx", sys.modules)

    def test_kernel_reuse_and_close_order(self):
        custom_ops = ctypes.CDLL(str(LIBRARY))
        for name in ("CustomOpLiveKernels", "CustomOpCreatedKernels"):
            getattr(custom_ops, name).argtypes = []
            getattr(custom_ops, name).restype = ctypes.c_int
        live = custom_ops.CustomOpLiveKernels()
        created = custom_ops.CustomOpCreatedKernels()
        session = make_session()
        values = numpy.array([-2, 1], dtype=numpy.float32)
        for _ in range(3):
            session.run(None, {"X": values})
        self.assertEqual(custom_ops.CustomOpCreatedKernels(), created + 1)
        self.assertEqual(custom_ops.CustomOpLiveKernels(), live + 1)
        session.close()
        session.close()
        self.assertEqual(custom_ops.CustomOpLiveKernels(), live)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.run(None, {"X": values})

    def test_sessions_and_attributes_are_independent(self):
        with make_session(make_model(0.25)) as first, make_session(make_model(0.5)) as second:
            values = numpy.array([-2, 1], dtype=numpy.float32)
            numpy.testing.assert_array_equal(first.run(None, {"X": values})[0], [-0.5, 1])
            numpy.testing.assert_array_equal(second.run(None, {"X": values})[0], [-1, 1])
            first.close()
            numpy.testing.assert_array_equal(second.run(None, {"X": values})[0], [-1, 1])

    def test_invalid_attributes_propagate(self):
        for alpha in (None, "not-a-float"):
            with self.subTest(alpha=alpha), make_session(make_model(alpha)) as session:
                with self.assertRaisesRegex(RuntimeError, "[Aa]ttribute"):
                    session.run(None, {"X": numpy.ones(2, numpy.float32)})

    def test_rejects_wrong_dtype_and_names(self):
        with make_session() as session:
            with self.assertRaisesRegex(TypeError, "float32"):
                session.run(None, {"X": numpy.ones(2, numpy.float64)})
            with self.assertRaisesRegex(ValueError, "input"):
                session.run(None, {"wrong": numpy.ones(2, numpy.float32)})
            with self.assertRaisesRegex(ValueError, "output"):
                session.run(["wrong"], {"X": numpy.ones(2, numpy.float32)})
        with self.assertRaisesRegex(RuntimeError, "float32"):
            make_session(make_model(dtype=TensorProto.DOUBLE))

    def test_serialized_model_and_noncontiguous_input(self):
        with make_session(make_model().SerializeToString()) as session:
            values = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)[:, ::2]
            numpy.testing.assert_array_equal(
                session.run(["Y"], {"X": values})[0],
                numpy.where(values < 0, values * 0.125, values),
            )

    def test_declared_shapes_are_checked_on_each_run(self):
        model = make_model()
        for value in (model.graph.input[0], model.graph.output[0]):
            value.type.tensor_type.shape.dim.add().dim_value = 2
        with make_session(model) as session:
            with self.assertRaisesRegex(RuntimeError, "[Ss]hape"):
                session.run(None, {"X": numpy.ones(3, numpy.float32)})
            result = session.run(None, {"X": numpy.ones(2, numpy.float32)})[0]
            numpy.testing.assert_array_equal(result, [1, 1])

    def test_unsupported_opset_missing_symbol_and_api(self):
        with self.assertRaisesRegex(RuntimeError, "opset"):
            make_session(make_model(version=2))
        with self.assertRaisesRegex(RuntimeError, "RegisterCustomOps"):
            make_session(library=HOST)
        with self.assertRaisesRegex(RuntimeError, "API 19"):
            make_session(library=BAD_LIBRARY)

    def test_rejects_unsupported_graphs(self):
        model = make_model()
        model.graph.node.append(model.graph.node[0])
        with self.assertRaisesRegex(RuntimeError, "single-node"):
            make_session(model)
        for empty_model in (ModelProto(), b""):
            with self.assertRaisesRegex(RuntimeError, "nonempty"):
                make_session(empty_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--bad-library", type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    HOST = arguments.host.resolve(strict=True)
    LIBRARY = arguments.library.resolve(strict=True)
    BAD_LIBRARY = arguments.bad_library.resolve(strict=True)
    unittest.main(argv=[__file__, *remaining], verbosity=2)
