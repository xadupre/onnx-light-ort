// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_ort/common_light_to_ort/leaky_relu_kernel.h"
#include "onnx_light_ort/common_light_to_ort/unary_float32.h"

namespace onnx_light_ort::common_light_to_ort {

OrtStatus *LeakyReluKernel::Compute(OrtKernelContext *context) const {
  using onnx_light::core::runtime::DataType;
  using onnx_light::core::runtime::Tensor;
  UnaryFloat32Buffers buffers;
  if (OrtStatus *status = GetUnaryFloat32Buffers(api, context, "LeakyRelu", buffers))
    return status;
  const size_t bytes = buffers.count * sizeof(float);
  const Tensor x = Tensor::Borrow("X", static_cast<int32_t>(DataType::FLOAT), buffers.shape,
                                  reinterpret_cast<const uint8_t *>(buffers.input), bytes);
  Tensor y = Tensor::Borrow("Y", static_cast<int32_t>(DataType::FLOAT), buffers.shape,
                            reinterpret_cast<const uint8_t *>(buffers.output), bytes);
  kernel(x, alpha, y);
  return nullptr;
}

} // namespace onnx_light_ort::common_light_to_ort
