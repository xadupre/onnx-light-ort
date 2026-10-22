"""
Custom ops: ORT called by onnx-light
====================================

The standalone onnx-light host loads an independent ORT ``LeakyRelu`` custom-op library,
compiled using only the public ORT headers and its own numerical code.
This script does not import or load ONNX Runtime.
"""

import os
from pathlib import Path

import numpy
from onnx_light.onnx import TensorProto, checker, helper

from onnx_light_ort import CustomOpSession

# %%
# This first host profile accepts one float32 CPU custom node. Unsupported
# signatures, opsets and API versions fail explicitly.
model = helper.make_model(
    helper.make_graph(
        [helper.make_node("LeakyRelu", ["X"], ["Y"], domain="ort_custom_ops", alpha=0.125)],
        "ort-custom-op-in-onnx-light",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N"])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N"])],
    ),
    opset_imports=[helper.make_opsetid("ort_custom_ops", 1)],
    ir_version=9,
)
checker.check_model(model)

# %%
# The library path is caller-owned: native plugins execute trusted code and
# are never discovered from the model. Kernels are retained between runs.
host = Path(os.environ["ONNX_LIGHT_ORT_HOST_LIBRARY"]).resolve(strict=True)
library = Path(os.environ["ONNX_LIGHT_ORT_INCOMING_CUSTOM_OP_LIBRARY"]).resolve(strict=True)
values = numpy.array([-4, -1, 0, 2], dtype=numpy.float32)
with CustomOpSession(model, library=library, host_library=host) as session:
    result = session.run(["Y"], {"X": values})[0]
    numpy.testing.assert_array_equal(result, [-0.5, -0.125, 0, 2])
    other = numpy.array([1, -2], dtype=numpy.float32)
    numpy.testing.assert_array_equal(
        session.run(None, {"X": other})[0], numpy.where(other < 0, other * 0.125, other)
    )
print("ORT custom op called by onnx-light:", result)
