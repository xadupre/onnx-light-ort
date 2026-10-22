// SPDX-License-Identifier: Apache-2.0

#ifndef ONNX_LIGHT_ORT_TESTS_ORT_TEST_HOST_H
#define ONNX_LIGHT_ORT_TESTS_ORT_TEST_HOST_H

#include "onnxruntime_c_api.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

struct OrtStatus {
  OrtErrorCode code;
  std::string message;
};
struct OrtTensorTypeAndShapeInfo {
  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  std::vector<int64_t> dimensions;
};
struct OrtMemoryInfo {
  OrtMemoryInfoDeviceType device = OrtMemoryInfoDeviceType_CPU;
};
struct OrtValue {
  OrtTensorTypeAndShapeInfo info;
  OrtMemoryInfo memory;
  void *data = nullptr;
};
struct OrtKernelContext {
  const OrtValue *input;
  OrtValue *output;
  OrtStatus *input_error = nullptr;
};
struct OrtKernelInfo {
  std::optional<float> alpha;
  bool wrong_type = false;
};
struct OrtCustomOpDomain {
  std::string name;
  std::vector<const OrtCustomOp *> ops;
};
struct OrtSessionOptions {
  std::vector<OrtCustomOpDomain *> domains;
  bool reject_registration = false;
};

namespace onnx_light_ort::test {

inline int live_shape_infos = 0;

inline void Require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

inline OrtApi MakeApi() {
  OrtApi api{};
  api.CreateStatus = [](OrtErrorCode code, const char *message) noexcept {
    return new OrtStatus{code, message};
  };
  api.ReleaseStatus = [](OrtStatus *status) noexcept { delete status; };
  api.GetErrorCode = [](const OrtStatus *status) noexcept { return status->code; };
  api.GetErrorMessage = [](const OrtStatus *status) noexcept { return status->message.c_str(); };
  api.CreateCustomOpDomain = [](const char *name, OrtCustomOpDomain **out) noexcept -> OrtStatus * {
    *out = new OrtCustomOpDomain{name, {}};
    return nullptr;
  };
  api.ReleaseCustomOpDomain = [](OrtCustomOpDomain *domain) noexcept { delete domain; };
  api.CustomOpDomain_Add = [](OrtCustomOpDomain *domain,
                              const OrtCustomOp *op) noexcept -> OrtStatus * {
    domain->ops.push_back(op);
    return nullptr;
  };
  api.AddCustomOpDomain = [](OrtSessionOptions *options,
                             OrtCustomOpDomain *domain) noexcept -> OrtStatus * {
    if (options->reject_registration)
      return new OrtStatus{ORT_FAIL, "Registration rejected by the test host."};
    options->domains.push_back(domain);
    return nullptr;
  };
  api.KernelInfoGetAttribute_float = [](const OrtKernelInfo *info, const char *name,
                                        float *out) noexcept -> OrtStatus * {
    if (std::string(name) != "alpha" || !info->alpha || info->wrong_type)
      return new OrtStatus{ORT_INVALID_ARGUMENT, "Missing or invalid alpha."};
    *out = *info->alpha;
    return nullptr;
  };
  api.KernelContext_GetInput = [](const OrtKernelContext *context, size_t index,
                                  const OrtValue **out) noexcept -> OrtStatus * {
    if (context->input_error != nullptr)
      return context->input_error;
    if (index != 0)
      return new OrtStatus{ORT_INVALID_ARGUMENT, "Unexpected input index."};
    *out = context->input;
    return nullptr;
  };
  api.KernelContext_GetOutput = [](OrtKernelContext *context, size_t index, const int64_t *dims,
                                   size_t rank, OrtValue **out) noexcept -> OrtStatus * {
    if (index != 0)
      return new OrtStatus{ORT_INVALID_ARGUMENT, "Unexpected output index."};
    *out = context->output;
    if (*out != nullptr && (*out)->info.dimensions != std::vector<int64_t>(dims, dims + rank))
      return new OrtStatus{ORT_INVALID_ARGUMENT, "Output shape differs from the requested shape."};
    return nullptr;
  };
  api.GetTensorMemoryInfo = [](const OrtValue *value,
                               const OrtMemoryInfo **out) noexcept -> OrtStatus * {
    *out = &value->memory;
    return nullptr;
  };
  api.MemoryInfoGetDeviceType = [](const OrtMemoryInfo *info,
                                   OrtMemoryInfoDeviceType *out) noexcept { *out = info->device; };
  api.GetTensorTypeAndShape = [](const OrtValue *value,
                                 OrtTensorTypeAndShapeInfo **out) noexcept -> OrtStatus * {
    *out = new OrtTensorTypeAndShapeInfo(value->info);
    ++live_shape_infos;
    return nullptr;
  };
  api.ReleaseTensorTypeAndShapeInfo = [](OrtTensorTypeAndShapeInfo *info) noexcept {
    delete info;
    --live_shape_infos;
  };
  api.GetTensorElementType = [](const OrtTensorTypeAndShapeInfo *info,
                                ONNXTensorElementDataType *out) noexcept -> OrtStatus * {
    *out = info->type;
    return nullptr;
  };
  api.GetDimensionsCount = [](const OrtTensorTypeAndShapeInfo *info,
                              size_t *out) noexcept -> OrtStatus * {
    *out = info->dimensions.size();
    return nullptr;
  };
  api.GetDimensions = [](const OrtTensorTypeAndShapeInfo *info, int64_t *out,
                         size_t count) noexcept -> OrtStatus * {
    if (count != info->dimensions.size())
      return new OrtStatus{ORT_INVALID_ARGUMENT, "Unexpected rank."};
    std::copy(info->dimensions.begin(), info->dimensions.end(), out);
    return nullptr;
  };
  api.GetTensorMutableData = [](OrtValue *value, void **out) noexcept -> OrtStatus * {
    *out = value->data;
    return nullptr;
  };
  return api;
}

inline const OrtApi api = MakeApi();
inline const OrtApiBase api_base{
    [](uint32_t version) noexcept -> const OrtApi * {
      return version >= 1 && version <= 18 ? &api : nullptr;
    },
    []() noexcept { return "onnx-light-ort explicit unit-test host"; }};

inline void Check(OrtStatus *status) {
  if (status != nullptr) {
    const std::string message = status->message;
    api.ReleaseStatus(status);
    throw std::runtime_error(message);
  }
}

inline void RequireError(OrtStatus *status, OrtErrorCode expected) {
  Require(status != nullptr, "Expected an error status.");
  const OrtErrorCode actual = status->code;
  api.ReleaseStatus(status);
  Require(actual == expected, "Unexpected error code.");
}

} // namespace onnx_light_ort::test

#endif
