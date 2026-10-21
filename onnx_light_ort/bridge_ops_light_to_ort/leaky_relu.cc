// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_ort/bridge_ops_light_to_ort/leaky_relu.h"
#include "onnx_light_ort/common_light_to_ort/api_status.h"
#include "onnx_light_ort/common_light_to_ort/leaky_relu_kernel.h"

namespace onnx_light_ort::bridge_ops_light_to_ort {

using common_light_to_ort::LeakyReluKernel;

OrtStatus *ORT_API_CALL CreateLeakyRelu(const OrtCustomOp *, const OrtApi *api,
                                        const OrtKernelInfo *info, void **result) noexcept {
  *result = nullptr;
  return common_light_to_ort::Boundary(*api, [&]() -> OrtStatus * {
    float alpha = 0;
    if (OrtStatus *status = api->KernelInfoGetAttribute_float(info, "alpha", &alpha))
      return status;
    *result = new LeakyReluKernel{*api, alpha};
    return nullptr;
  });
}

OrtStatus *ORT_API_CALL ComputeLeakyRelu(void *state, OrtKernelContext *context) noexcept {
  const auto &kernel = *static_cast<LeakyReluKernel *>(state);
  return common_light_to_ort::Boundary(kernel.api, [&] { return kernel.Compute(context); });
}

void ORT_API_CALL DestroyLeakyRelu(void *state) noexcept {
  delete static_cast<LeakyReluKernel *>(state);
}

} // namespace onnx_light_ort::bridge_ops_light_to_ort
