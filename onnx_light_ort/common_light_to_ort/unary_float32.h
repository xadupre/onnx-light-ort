// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnxruntime_c_api.h"

namespace onnx_light_ort::common_light_to_ort {

struct UnaryFloat32Buffers {
  onnx_light::core::runtime::Shape shape;
  const float *input = nullptr;
  float *output = nullptr;
  size_t count = 0;
};

OrtStatus *GetUnaryFloat32Buffers(const OrtApi &api, OrtKernelContext *context, const char *name,
                                  UnaryFloat32Buffers &buffers);

} // namespace onnx_light_ort::common_light_to_ort
