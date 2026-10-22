"""Runs the onnx-light CPU custom op in a real ONNX Runtime session."""

import argparse
from pathlib import Path

import numpy
import onnxruntime
from onnx_light.onnx import TensorProto, checker, helper, save


def make_model():
    """Creates a model explicitly calling the onnx-light custom operator."""
    graph = helper.make_graph(
        [helper.make_node("LeakyRelu", ["X"], ["Y"], domain="onnx_light", alpha=0.125)],
        "custom-ops-example",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("onnx_light", 1),
        ],
        ir_version=9,
    )


def main():
    """Runs the native custom op in real ORT and optionally saves its model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model-output", type=Path)
    args = parser.parse_args()
    model = make_model()
    checker.check_model(model)
    if args.model_output is not None:
        save(model, args.model_output)
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.register_custom_ops_library(str(args.library.resolve(strict=True)))
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    values = numpy.array([-4, -1, 0, 2], dtype=numpy.float32)
    result = session.run(["Y"], {"X": values})[0]
    numpy.testing.assert_array_equal(result, numpy.array([-0.5, -0.125, 0, 2], numpy.float32))
    print(result)


if __name__ == "__main__":
    main()
