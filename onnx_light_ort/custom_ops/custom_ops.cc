// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_ort/bridge_ops_light_to_ort/leaky_relu.h"
#include "onnx_light_ort/common_light_to_ort/api_status.h"
#include "onnxruntime_c_api.h"

#include <memory>
#include <mutex>
#include <vector>

#ifdef _WIN32
#define ONNX_LIGHT_ORT_EXPORT __declspec(dllexport)
#else
#define ONNX_LIGHT_ORT_EXPORT __attribute__((visibility("default")))
#endif

namespace {

constexpr uint32_t kApiVersion = 18;

#define RETURN_IF_ERROR(expression)                                                                \
  do {                                                                                             \
    if (OrtStatus *status = (expression))                                                          \
      return status;                                                                               \
  } while (false)

using onnx_light_ort::common_light_to_ort::Boundary;

struct CustomOp final : OrtCustomOp {
  CustomOp() : OrtCustomOp{} {
    version = kApiVersion;
    GetName = [](const OrtCustomOp *) { return "LeakyRelu"; };
    GetExecutionProviderType = [](const OrtCustomOp *) { return "CPUExecutionProvider"; };
    GetInputTypeCount = GetOutputTypeCount = [](const OrtCustomOp *) -> size_t { return 1; };
    GetInputType = GetOutputType = [](const OrtCustomOp *, size_t) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    };
    GetInputCharacteristic =
        GetOutputCharacteristic = [](const OrtCustomOp *, size_t) { return INPUT_OUTPUT_REQUIRED; };
    GetInputMemoryType = [](const OrtCustomOp *, size_t) { return OrtMemTypeCPUInput; };
    GetStartVersion = GetEndVersion = [](const OrtCustomOp *) { return 1; };
    CreateKernelV2 = onnx_light_ort::bridge_ops_light_to_ort::CreateLeakyRelu;
    KernelComputeV2 = onnx_light_ort::bridge_ops_light_to_ort::ComputeLeakyRelu;
    KernelDestroy = onnx_light_ort::bridge_ops_light_to_ort::DestroyLeakyRelu;
  }
};

struct Domain {
  const OrtApi &api;
  CustomOp op;
  OrtCustomOpDomain *handle = nullptr;

  ~Domain() {
    if (handle != nullptr)
      api.ReleaseCustomOpDomain(handle);
  }
};

struct Domains {
  std::mutex mutex;
  std::vector<std::unique_ptr<Domain>> values;
};

} // namespace

extern "C" ONNX_LIGHT_ORT_EXPORT OrtStatus *ORT_API_CALL
RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api_base) {
  const OrtApi *api = api_base->GetApi(kApiVersion);
  if (api == nullptr)
    return api_base->GetApi(1)->CreateStatus(ORT_NOT_IMPLEMENTED,
                                             "onnx-light custom ops require ORT API 18.");
  return Boundary(*api, [&]() -> OrtStatus * {
    static Domains domains;
    const std::lock_guard<std::mutex> lock(domains.mutex);
    for (const auto &domain : domains.values) {
      if (&domain->api == api)
        return api->AddCustomOpDomain(options, domain->handle);
    }
    auto domain = std::make_unique<Domain>(*api);
    RETURN_IF_ERROR(api->CreateCustomOpDomain("onnx_light", &domain->handle));
    RETURN_IF_ERROR(api->CustomOpDomain_Add(domain->handle, &domain->op));
    domains.values.push_back(std::move(domain));
    return api->AddCustomOpDomain(options, domains.values.back()->handle);
  });
}
