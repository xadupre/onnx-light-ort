// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnxruntime_c_api.h"

#include <array>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
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

template <typename Function> OrtStatus *Boundary(const OrtApi &api, Function &&function) noexcept {
  try {
    return function();
  } catch (const std::invalid_argument &error) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, error.what());
  } catch (const std::exception &error) {
    return api.CreateStatus(ORT_RUNTIME_EXCEPTION, error.what());
  } catch (...) {
    return api.CreateStatus(ORT_RUNTIME_EXCEPTION, "Unknown exception in onnx-light custom op.");
  }
}

struct ShapeInfoDeleter {
  const OrtApi &api;
  void operator()(OrtTensorTypeAndShapeInfo *value) const noexcept {
    api.ReleaseTensorTypeAndShapeInfo(value);
  }
};

struct LeakyReluKernel {
  const OrtApi &api;
  const float alpha;
  onnx_light::onnx_kernels::kernel::LeakyRelu kernel{
      onnx_light::core::runtime::KernelContext{onnx_light::core::runtime::DefaultOpset(16)}};

  OrtStatus *Compute(OrtKernelContext *context) const {
    using onnx_light::core::runtime::DataType;
    using onnx_light::core::runtime::Shape;
    using onnx_light::core::runtime::Tensor;

    const OrtValue *input = nullptr;
    RETURN_IF_ERROR(api.KernelContext_GetInput(context, 0, &input));
    if (input == nullptr)
      return api.CreateStatus(ORT_INVALID_ARGUMENT, "LeakyRelu requires input X.");

    const OrtMemoryInfo *memory = nullptr;
    RETURN_IF_ERROR(api.GetTensorMemoryInfo(input, &memory));
    OrtMemoryInfoDeviceType device;
    api.MemoryInfoGetDeviceType(memory, &device);
    if (device != OrtMemoryInfoDeviceType_CPU)
      return api.CreateStatus(ORT_INVALID_ARGUMENT, "LeakyRelu requires CPU memory.");

    OrtTensorTypeAndShapeInfo *raw_info = nullptr;
    RETURN_IF_ERROR(api.GetTensorTypeAndShape(input, &raw_info));
    const std::unique_ptr<OrtTensorTypeAndShapeInfo, ShapeInfoDeleter> info(raw_info, {api});
    ONNXTensorElementDataType type;
    RETURN_IF_ERROR(api.GetTensorElementType(info.get(), &type));
    if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      return api.CreateStatus(ORT_INVALID_ARGUMENT, "LeakyRelu requires a float32 tensor.");
    size_t rank = 0;
    RETURN_IF_ERROR(api.GetDimensionsCount(info.get(), &rank));
    if (rank > Shape::kMaxRank)
      return api.CreateStatus(ORT_INVALID_ARGUMENT, "LeakyRelu supports at most 16 dimensions.");
    std::array<int64_t, Shape::kMaxRank> dimensions{};
    RETURN_IF_ERROR(api.GetDimensions(info.get(), dimensions.data(), rank));
    Shape shape;
    for (size_t i = 0; i < rank; ++i)
      shape.push_back(dimensions[i]);
    const int64_t elements = shape.product(0, rank, "ORT LeakyRelu input");
    const size_t bytes =
        onnx_light::core::runtime::PackedByteSize(static_cast<int32_t>(DataType::FLOAT), elements);

    void *input_data = nullptr;
    // API 18 exposes tensor data through a mutable accessor even for read-only inputs.
    RETURN_IF_ERROR(api.GetTensorMutableData(const_cast<OrtValue *>(input), &input_data));
    OrtValue *output = nullptr;
    RETURN_IF_ERROR(api.KernelContext_GetOutput(context, 0, dimensions.data(), rank, &output));
    if (output == nullptr)
      return api.CreateStatus(ORT_INVALID_ARGUMENT, "LeakyRelu requires output Y.");
    void *output_data = nullptr;
    RETURN_IF_ERROR(api.GetTensorMutableData(output, &output_data));
    if (bytes != 0 && (input_data == nullptr || output_data == nullptr))
      return api.CreateStatus(ORT_INVALID_ARGUMENT, "LeakyRelu received a null tensor buffer.");
    if (bytes != 0 && (reinterpret_cast<uintptr_t>(input_data) % alignof(float) != 0 ||
                       reinterpret_cast<uintptr_t>(output_data) % alignof(float) != 0))
      return api.CreateStatus(ORT_INVALID_ARGUMENT, "LeakyRelu requires float-aligned buffers.");
    const uintptr_t source = reinterpret_cast<uintptr_t>(input_data);
    const uintptr_t destination = reinterpret_cast<uintptr_t>(output_data);
    if (bytes != 0 &&
        (source <= destination ? destination - source < bytes : source - destination < bytes))
      return api.CreateStatus(ORT_INVALID_ARGUMENT, "LeakyRelu requires non-overlapping buffers.");

    const Tensor x = Tensor::Borrow("X", static_cast<int32_t>(DataType::FLOAT), shape,
                                    static_cast<const uint8_t *>(input_data), bytes);
    Tensor y = Tensor::Borrow("Y", static_cast<int32_t>(DataType::FLOAT), shape,
                              static_cast<const uint8_t *>(output_data), bytes);
    kernel(x, alpha, y);
    return nullptr;
  }
};

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
    CreateKernelV2 = [](const OrtCustomOp *, const OrtApi *api, const OrtKernelInfo *info,
                        void **result) -> OrtStatus * {
      *result = nullptr;
      return Boundary(*api, [&]() -> OrtStatus * {
        float alpha = 0;
        RETURN_IF_ERROR(api->KernelInfoGetAttribute_float(info, "alpha", &alpha));
        *result = new LeakyReluKernel{*api, alpha};
        return nullptr;
      });
    };
    KernelComputeV2 = [](void *state, OrtKernelContext *context) -> OrtStatus * {
      const auto &kernel = *static_cast<LeakyReluKernel *>(state);
      return Boundary(kernel.api, [&] { return kernel.Compute(context); });
    };
    KernelDestroy = [](void *state) { delete static_cast<LeakyReluKernel *>(state); };
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
