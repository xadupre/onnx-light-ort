// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_ort/common_light_to_ort/log_kernel.h"
#include "onnx_light_ort/common_light_to_ort/unary_tensor.h"

#include <ml_cpu_compute/math/math.h>

namespace onnx_light_ort::common_light_to_ort {

OrtStatus *LogKernel::Compute(OrtKernelContext *context) const {
  UnaryTensorBuffers buffers;
  if (OrtStatus *status = GetUnaryTensorBuffers(api, context, "Log", buffers))
    return status;
  switch (buffers.type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    ml_cpu_compute::LogFloat32(static_cast<const float *>(buffers.input),
                               static_cast<float *>(buffers.output), buffers.count);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    ml_cpu_compute::LogFloat64(static_cast<const double *>(buffers.input),
                               static_cast<double *>(buffers.output), buffers.count);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    ml_cpu_compute::LogFloat16(static_cast<const uint16_t *>(buffers.input),
                               static_cast<uint16_t *>(buffers.output), buffers.count);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    ml_cpu_compute::LogBFloat16WithTuning(static_cast<const uint16_t *>(buffers.input),
                                          static_cast<uint16_t *>(buffers.output), buffers.count,
                                          ml_cpu_compute::kDefaultLogBFloat16ExecutionTuning);
    break;
  default:
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "Unsupported Log tensor type.");
  }
  return nullptr;
}

} // namespace onnx_light_ort::common_light_to_ort
