// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_proto/onnx.h"
#include "onnxruntime_c_api.h"

#include <optional>
#include <string>
#include <vector>

struct OrtStatus {
  OrtErrorCode code;
  std::string message;
};

struct OrtTensorTypeAndShapeInfo {
  onnx_light::core::runtime::Shape shape;
};

struct OrtMemoryInfo {};

struct OrtValue {
  const onnx_light::core::runtime::Tensor *tensor = nullptr;
};

struct OrtKernelInfo {
  const onnx_light::NodeProto *node;
};

struct OrtKernelContext {
  OrtValue input;
  OrtValue output;
  std::optional<onnx_light::core::runtime::Tensor> tensor;
};

struct OrtCustomOpDomain {
  std::string name;
  std::vector<const OrtCustomOp *> ops;
};

struct OrtSessionOptions {
  std::vector<OrtCustomOpDomain *> domains;
};

namespace onnx_light_ort::common_ort_to_light {

const OrtApi &HostApi();
const OrtApiBase &HostApiBase();
void Check(OrtStatus *status);
size_t TensorBytes(const onnx_light::core::runtime::Shape &shape);

} // namespace onnx_light_ort::common_ort_to_light
