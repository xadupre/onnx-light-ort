"""
Custom ops: onnx-light called by ORT
====================================

The real ORT host loads our native ``LeakyRelu`` custom-op library.
The model calls the custom domain explicitly; no EP is involved.
"""

import os
from pathlib import Path

import numpy
import onnxruntime
from onnx_light.onnx import TensorProto, checker, helper

# %%
# Create a model calling the custom domain ``onnx_light``.
model = helper.make_model(
    helper.make_graph(
        [helper.make_node("LeakyRelu", ["X"], ["Y"], domain="onnx_light", alpha=0.125)],
        "onnx-light-in-ort",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])],
    ),
    opset_imports=[
        helper.make_opsetid("", 18),
        helper.make_opsetid("onnx_light", 1),
    ],
    ir_version=9,
)
checker.check_model(model)

# %%
# ORT supplies its API table and allocates the outputs. The plugin delegates
# the numerical work to onnx-light's native CPU kernel.
library = Path(os.environ["ONNX_LIGHT_ORT_PLUGIN_LIBRARY"]).resolve(strict=True)
options = onnxruntime.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.register_custom_ops_library(str(library))
session = onnxruntime.InferenceSession(
    model.SerializeToString(), options, providers=["CPUExecutionProvider"]
)
values = numpy.array([-4, -1, 0, 2], dtype=numpy.float32)
result = session.run(["Y"], {"X": values})[0]
numpy.testing.assert_array_equal(result, [-0.5, -0.125, 0, 2])
print("onnx-light custom op called by ORT:", result)
