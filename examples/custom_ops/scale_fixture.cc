// SPDX-License-Identifier: Apache-2.0

#include "onnxruntime_c_api.h"

#include <array>
#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

#ifdef _WIN32
#define FIXTURE_EXPORT __declspec(dllexport)
#else
#define FIXTURE_EXPORT __attribute__((visibility("default")))
#endif

namespace {

std::atomic<int> live{0};
std::atomic<int> created{0};

struct Kernel {
  const OrtApi *api;
  float scale;
};

struct Scale : OrtCustomOp {
  Scale() : OrtCustomOp{} {
    version = 18;
    GetName = [](const OrtCustomOp *) { return "Scale"; };
    GetExecutionProviderType = [](const OrtCustomOp *) -> const char * { return nullptr; };
    GetInputTypeCount = GetOutputTypeCount = [](const OrtCustomOp *) -> size_t { return 1; };
    GetInputType = GetOutputType = [](const OrtCustomOp *, size_t) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    };
    GetInputCharacteristic =
        GetOutputCharacteristic = [](const OrtCustomOp *, size_t) { return INPUT_OUTPUT_REQUIRED; };
    GetInputMemoryType = [](const OrtCustomOp *, size_t) { return OrtMemTypeDefault; };
    GetStartVersion = GetEndVersion = [](const OrtCustomOp *) { return 1; };
    CreateKernelV2 = [](const OrtCustomOp *, const OrtApi *api, const OrtKernelInfo *info,
                        void **out) -> OrtStatus * {
      float scale = 0;
      if (OrtStatus *status = api->KernelInfoGetAttribute_float(info, "scale", &scale))
        return status;
      auto *kernel = new (std::nothrow) Kernel{api, scale};
      if (kernel == nullptr)
        return api->CreateStatus(ORT_FAIL, "Cannot allocate Scale kernel.");
      *out = kernel;
      ++live;
      ++created;
      return nullptr;
    };
    KernelDestroy = [](void *state) {
      delete static_cast<Kernel *>(state);
      --live;
    };
    KernelComputeV2 = [](void *state, OrtKernelContext *context) -> OrtStatus * {
      const auto &kernel = *static_cast<Kernel *>(state);
      const OrtApi &api = *kernel.api;
      const OrtValue *input = nullptr;
      OrtStatus *status = api.KernelContext_GetInput(context, 0, &input);
      if (status)
        return status;
      OrtTensorTypeAndShapeInfo *info = nullptr;
      status = api.GetTensorTypeAndShape(input, &info);
      if (status)
        return status;
      size_t rank = 0;
      status = api.GetDimensionsCount(info, &rank);
      std::array<int64_t, 16> dimensions{};
      size_t count = 0;
      if (!status && rank > dimensions.size())
        status = api.CreateStatus(ORT_INVALID_ARGUMENT, "Scale supports at most 16 dimensions.");
      if (!status)
        status = api.GetDimensions(info, dimensions.data(), rank);
      if (!status)
        status = api.GetTensorShapeElementCount(info, &count);
      api.ReleaseTensorTypeAndShapeInfo(info);
      if (status)
        return status;
      OrtValue *output = nullptr;
      status = api.KernelContext_GetOutput(context, 0, dimensions.data(), rank, &output);
      if (status)
        return status;
      void *input_data = nullptr;
      void *output_data = nullptr;
      status = api.GetTensorMutableData(const_cast<OrtValue *>(input), &input_data);
      if (!status)
        status = api.GetTensorMutableData(output, &output_data);
      if (status)
        return status;
      for (size_t i = 0; i < count; ++i)
        static_cast<float *>(output_data)[i] =
            kernel.scale * static_cast<const float *>(input_data)[i];
      return nullptr;
    };
  }
};

struct Domain {
  const OrtApi *api = nullptr;
  Scale op;
  OrtCustomOpDomain *handle = nullptr;
  ~Domain() {
    if (handle)
      api->ReleaseCustomOpDomain(handle);
  }
};

} // namespace

extern "C" FIXTURE_EXPORT OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                                                    const OrtApiBase *base) {
  const OrtApi *api = base->GetApi(18);
  if (api == nullptr)
    return base->GetApi(1)->CreateStatus(ORT_NOT_IMPLEMENTED, "Scale requires API 18.");
#ifdef FIXTURE_BAD_VERSION
  if (base->GetApi(19) == nullptr)
    return api->CreateStatus(ORT_NOT_IMPLEMENTED, "Fixture requires unavailable API 19.");
#endif
  try {
    static std::mutex mutex;
    static std::vector<std::unique_ptr<Domain>> domains;
    const std::lock_guard<std::mutex> lock(mutex);
    for (const auto &domain : domains) {
      if (domain->api == api)
        return api->AddCustomOpDomain(options, domain->handle);
    }
    auto domain = std::make_unique<Domain>();
    domain->api = api;
    if (OrtStatus *status = api->CreateCustomOpDomain("ort_fixture", &domain->handle))
      return status;
    if (OrtStatus *status = api->CustomOpDomain_Add(domain->handle, &domain->op))
      return status;
    domains.push_back(std::move(domain));
    return api->AddCustomOpDomain(options, domains.back()->handle);
  } catch (const std::exception &error) {
    return api->CreateStatus(ORT_FAIL, error.what());
  }
}

extern "C" FIXTURE_EXPORT int FixtureLiveKernels() { return live.load(); }
extern "C" FIXTURE_EXPORT int FixtureCreatedKernels() { return created.load(); }
