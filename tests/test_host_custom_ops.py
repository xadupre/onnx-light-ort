"""Checks real plugin execution in onnx-light without importing ONNX Runtime."""

import argparse
import ctypes
from pathlib import Path
import sys
import unittest

import numpy
import onnx

from onnx_light_ort import CustomOpSession

HOST = None
LIBRARY = None
BAD_LIBRARY = None


def make_model(scale=3.0, *, version=1, dtype=onnx.TensorProto.FLOAT):
    """Creates a single-node model for the independent ORT fixture."""
    attributes = {} if scale is None else {"scale": scale}
    return onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Scale", ["X"], ["Y"], domain="ort_fixture", **attributes)],
            "ort-plugin-in-onnx-light",
            [onnx.helper.make_tensor_value_info("X", dtype, None)],
            [onnx.helper.make_tensor_value_info("Y", dtype, None)],
        ),
        opset_imports=[onnx.helper.make_opsetid("ort_fixture", version)],
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
                        numpy.testing.assert_array_equal(result, values * numpy.float32(3))
                        numpy.testing.assert_array_equal(values, original)
                        self.assertEqual(result.shape, shape)
                        self.assertEqual(result.dtype, numpy.float32)
                        if previous is not None:
                            numpy.testing.assert_array_equal(previous, result)
                            self.assertFalse(numpy.shares_memory(previous, result))
                        previous = result
            self.assertNotIn("onnxruntime", sys.modules)

    def test_kernel_reuse_and_close_order(self):
        fixture = ctypes.CDLL(str(LIBRARY))
        for name in ("FixtureLiveKernels", "FixtureCreatedKernels"):
            getattr(fixture, name).argtypes = []
            getattr(fixture, name).restype = ctypes.c_int
        live = fixture.FixtureLiveKernels()
        created = fixture.FixtureCreatedKernels()
        session = make_session()
        values = numpy.array([-2, 1], dtype=numpy.float32)
        for _ in range(3):
            session.run(None, {"X": values})
        self.assertEqual(fixture.FixtureCreatedKernels(), created + 1)
        self.assertEqual(fixture.FixtureLiveKernels(), live + 1)
        session.close()
        session.close()
        self.assertEqual(fixture.FixtureLiveKernels(), live)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.run(None, {"X": values})

    def test_sessions_and_attributes_are_independent(self):
        with make_session(make_model(2.0)) as first, make_session(make_model(5.0)) as second:
            values = numpy.array([-2, 1], dtype=numpy.float32)
            numpy.testing.assert_array_equal(first.run(None, {"X": values})[0], [-4, 2])
            numpy.testing.assert_array_equal(second.run(None, {"X": values})[0], [-10, 5])
            first.close()
            numpy.testing.assert_array_equal(second.run(None, {"X": values})[0], [-10, 5])

    def test_invalid_attributes_propagate(self):
        for scale in (None, "not-a-float"):
            with self.subTest(scale=scale), make_session(make_model(scale)) as session:
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
            make_session(make_model(dtype=onnx.TensorProto.DOUBLE))

    def test_serialized_model_and_noncontiguous_input(self):
        with make_session(make_model().SerializeToString()) as session:
            values = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)[:, ::2]
            numpy.testing.assert_array_equal(session.run(["Y"], {"X": values})[0], values * 3)

    def test_declared_shapes_are_checked_on_each_run(self):
        model = make_model()
        for value in (model.graph.input[0], model.graph.output[0]):
            value.type.tensor_type.shape.dim.add().dim_value = 2
        with make_session(model) as session:
            with self.assertRaisesRegex(RuntimeError, "[Ss]hape"):
                session.run(None, {"X": numpy.ones(3, numpy.float32)})
            result = session.run(None, {"X": numpy.ones(2, numpy.float32)})[0]
            numpy.testing.assert_array_equal(result, [3, 3])

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
        with self.assertRaisesRegex(RuntimeError, "nonempty"):
            CustomOpSession(
                onnx.ModelProto().SerializeToString(), library=LIBRARY, host_library=HOST
            )


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
