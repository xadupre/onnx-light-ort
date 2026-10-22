// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnxruntime_c_api.h"

namespace onnx_light_ort::bridge_ops_light_called_by_ort {

OrtStatus *ORT_API_CALL CreateLog(const OrtCustomOp *, const OrtApi *api, const OrtKernelInfo *,
                                  void **result) noexcept;
OrtStatus *ORT_API_CALL ComputeLog(void *state, OrtKernelContext *context) noexcept;
void ORT_API_CALL DestroyLog(void *state) noexcept;

} // namespace onnx_light_ort::bridge_ops_light_called_by_ort
