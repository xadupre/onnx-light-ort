// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnxruntime_c_api.h"

namespace onnx_light_ort::common_light_to_ort {

struct UnaryTensorBuffers {
  onnx_light::core::runtime::Shape shape;
  const void *input = nullptr;
  void *output = nullptr;
  size_t count = 0;
  size_t bytes = 0;
  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
};

OrtStatus *GetUnaryTensorBuffers(const OrtApi &api, OrtKernelContext *context, const char *name,
                                 UnaryTensorBuffers &buffers);

} // namespace onnx_light_ort::common_light_to_ort
