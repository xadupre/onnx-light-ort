"""
An ORT custom-op library in onnx-light
==================================================

The standalone onnx-light host loads an independent ORT ``Scale`` fixture,
compiled using only the public ORT headers and its own numerical code.
This script does not import or load ONNX Runtime.
"""

import os
from pathlib import Path

import numpy
import onnx

from onnx_light_ort import CustomOpSession

# %%
# This first host profile accepts one float32 CPU custom node. Unsupported
# signatures, opsets and API versions fail explicitly.
model = onnx.helper.make_model(
    onnx.helper.make_graph(
        [onnx.helper.make_node("Scale", ["X"], ["Y"], domain="ort_fixture", scale=3.0)],
        "ort-custom-op-in-onnx-light",
        [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, ["N"])],
        [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, ["N"])],
    ),
    opset_imports=[onnx.helper.make_opsetid("ort_fixture", 1)],
    ir_version=9,
)
onnx.checker.check_model(model)

# %%
# The library path is caller-owned: native plugins execute trusted code and
# are never discovered from the model. Kernels are retained between runs.
host = Path(os.environ["ONNX_LIGHT_ORT_HOST_LIBRARY"]).resolve(strict=True)
library = Path(os.environ["ONNX_LIGHT_ORT_FIXTURE_LIBRARY"]).resolve(strict=True)
values = numpy.array([-4, -1, 0, 2], dtype=numpy.float32)
with CustomOpSession(model, library=library, host_library=host) as session:
    result = session.run(["Y"], {"X": values})[0]
    numpy.testing.assert_array_equal(result, [-12, -3, 0, 6])
    other = numpy.array([1, -2], dtype=numpy.float32)
    numpy.testing.assert_array_equal(session.run(None, {"X": other})[0], other * 3)
print("ORT custom op in onnx-light:", result)
