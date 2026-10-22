// SPDX-License-Identifier: Apache-2.0

#ifndef ONNX_LIGHT_ORT_BRIDGE_OPS_LIGHT_TO_ORT_LEAKY_RELU_H
#define ONNX_LIGHT_ORT_BRIDGE_OPS_LIGHT_TO_ORT_LEAKY_RELU_H

#include "onnxruntime_c_api.h"

namespace onnx_light_ort::bridge_ops_light_called_by_ort {

OrtStatus *ORT_API_CALL CreateLeakyRelu(const OrtCustomOp *, const OrtApi *api,
                                        const OrtKernelInfo *info, void **result) noexcept;
OrtStatus *ORT_API_CALL ComputeLeakyRelu(void *state, OrtKernelContext *context) noexcept;
void ORT_API_CALL DestroyLeakyRelu(void *state) noexcept;

} // namespace onnx_light_ort::bridge_ops_light_called_by_ort

#endif
