// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnxruntime_c_api.h"

namespace onnx_light_ort::common_light_to_ort {

struct LogKernel {
  const OrtApi &api;
  OrtStatus *Compute(OrtKernelContext *context) const;
};

} // namespace onnx_light_ort::common_light_to_ort
