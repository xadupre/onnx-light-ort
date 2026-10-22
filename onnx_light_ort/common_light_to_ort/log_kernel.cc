// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_ort/common_light_to_ort/log_kernel.h"
#include "onnx_light_ort/common_light_to_ort/unary_float32.h"

#include <ml_cpu_compute/math/math.h>

namespace onnx_light_ort::common_light_to_ort {

OrtStatus *LogKernel::Compute(OrtKernelContext *context) const {
  UnaryFloat32Buffers buffers;
  if (OrtStatus *status = GetUnaryFloat32Buffers(api, context, "Log", buffers))
    return status;
  ml_cpu_compute::LogFloat32(buffers.input, buffers.output, buffers.count);
  return nullptr;
}

} // namespace onnx_light_ort::common_light_to_ort
