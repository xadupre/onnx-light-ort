// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_ort/bridge_ops_light_called_by_ort/log.h"
#include "onnx_light_ort/common_light_to_ort/api_status.h"
#include "onnx_light_ort/common_light_to_ort/log_kernel.h"

namespace onnx_light_ort::bridge_ops_light_called_by_ort {

using common_light_to_ort::LogKernel;

OrtStatus *ORT_API_CALL CreateLog(const OrtCustomOp *, const OrtApi *api, const OrtKernelInfo *,
                                  void **result) noexcept {
  *result = nullptr;
  return common_light_to_ort::Boundary(*api, [&]() -> OrtStatus * {
    *result = new LogKernel{*api};
    return nullptr;
  });
}

OrtStatus *ORT_API_CALL ComputeLog(void *state, OrtKernelContext *context) noexcept {
  const auto &kernel = *static_cast<LogKernel *>(state);
  return common_light_to_ort::Boundary(kernel.api, [&] { return kernel.Compute(context); });
}

void ORT_API_CALL DestroyLog(void *state) noexcept { delete static_cast<LogKernel *>(state); }

} // namespace onnx_light_ort::bridge_ops_light_called_by_ort
