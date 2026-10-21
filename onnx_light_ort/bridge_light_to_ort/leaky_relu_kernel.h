// SPDX-License-Identifier: Apache-2.0

#ifndef ONNX_LIGHT_ORT_BRIDGE_LIGHT_TO_ORT_LEAKY_RELU_KERNEL_H
#define ONNX_LIGHT_ORT_BRIDGE_LIGHT_TO_ORT_LEAKY_RELU_KERNEL_H

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnxruntime_c_api.h"

namespace onnx_light_ort::bridge_light_to_ort {

/// Retains native kernel state while adapting invocation-scoped ORT buffers.
struct LeakyReluKernel {
  const OrtApi &api;
  const float alpha;
  onnx_light::onnx_kernels::kernel::LeakyRelu kernel{
      onnx_light::core::runtime::KernelContext{onnx_light::core::runtime::DefaultOpset(16)}};

  /// Executes the kernel; native exceptions propagate to the caller's ABI boundary.
  OrtStatus *Compute(OrtKernelContext *context) const;
};

} // namespace onnx_light_ort::bridge_light_to_ort

#endif
