// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_ort/common_light_to_ort/leaky_relu_kernel.h"
#include "onnx_light_ort/common_light_to_ort/unary_tensor.h"

namespace onnx_light_ort::common_light_to_ort {

OrtStatus *LeakyReluKernel::Compute(OrtKernelContext *context) const {
  using onnx_light::core::runtime::Tensor;
  UnaryTensorBuffers buffers;
  if (OrtStatus *status = GetUnaryTensorBuffers(api, context, "LeakyRelu", buffers))
    return status;
  const Tensor x = Tensor::Borrow("X", static_cast<int32_t>(buffers.type), buffers.shape,
                                  static_cast<const uint8_t *>(buffers.input), buffers.bytes);
  Tensor y = Tensor::Borrow("Y", static_cast<int32_t>(buffers.type), buffers.shape,
                            static_cast<const uint8_t *>(buffers.output), buffers.bytes);
  kernel(x, alpha, y);
  return nullptr;
}

} // namespace onnx_light_ort::common_light_to_ort
