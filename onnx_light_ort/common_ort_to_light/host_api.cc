// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_ort/common_ort_to_light/host_api.h"
#include "onnx_light_ort/api_status.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace onnx_light_ort::common_ort_to_light {
namespace {

using onnx_light::core::runtime::Shape;
using onnx_light::core::runtime::Tensor;
using onnx_light_ort::Boundary;

void Require(bool condition, const char *message) {
  if (!condition)
    throw std::invalid_argument(message);
}

Shape ReadShape(const int64_t *dimensions, size_t rank) {
  Require(rank <= Shape::kMaxRank, "CPU custom ops support at most 16 dimensions.");
  Require(rank == 0 || dimensions != nullptr, "Null dimension buffer.");
  Shape shape;
  for (size_t i = 0; i < rank; ++i)
    shape.push_back(dimensions[i]);
  shape.product(0, rank, "custom-op shape");
  return shape;
}

OrtApi MakeApi() {
  OrtApi api{};
  api.CreateStatus = [](OrtErrorCode code, const char *message) noexcept {
    return new OrtStatus{code, message};
  };
  api.ReleaseStatus = [](OrtStatus *status) noexcept { delete status; };
  api.GetErrorCode = [](const OrtStatus *status) noexcept { return status->code; };
  api.GetErrorMessage = [](const OrtStatus *status) noexcept { return status->message.c_str(); };
  api.CreateCustomOpDomain = [](const char *name, OrtCustomOpDomain **out) noexcept -> OrtStatus * {
    return Boundary(HostApi(), [&]() -> OrtStatus * {
      Require(name != nullptr && out != nullptr, "Null custom-op domain argument.");
      *out = new OrtCustomOpDomain{name, {}};
      return nullptr;
    });
  };
  api.ReleaseCustomOpDomain = [](OrtCustomOpDomain *domain) noexcept { delete domain; };
  api.CustomOpDomain_Add = [](OrtCustomOpDomain *domain,
                              const OrtCustomOp *op) noexcept -> OrtStatus * {
    return Boundary(HostApi(), [&]() -> OrtStatus * {
      Require(domain != nullptr && op != nullptr, "Null custom-op descriptor.");
      domain->ops.push_back(op);
      return nullptr;
    });
  };
  api.AddCustomOpDomain = [](OrtSessionOptions *options,
                             OrtCustomOpDomain *domain) noexcept -> OrtStatus * {
    return Boundary(HostApi(), [&]() -> OrtStatus * {
      Require(options != nullptr && domain != nullptr, "Null custom-op registration.");
      options->domains.push_back(domain);
      return nullptr;
    });
  };
  api.KernelInfoGetAttribute_float = [](const OrtKernelInfo *info, const char *name,
                                        float *out) noexcept -> OrtStatus * {
    return Boundary(HostApi(), [&]() -> OrtStatus * {
      Require(info != nullptr && name != nullptr && out != nullptr, "Null attribute argument.");
      for (const auto &attribute : info->node->attribute()) {
        if (attribute.name() == name) {
          Require(attribute.type() == onnx_light::AttributeProto::FLOAT,
                  "Custom-op attribute must be a float.");
          *out = attribute.f();
          return nullptr;
        }
      }
      throw std::invalid_argument(std::string("Missing float attribute: ") + name);
    });
  };
  api.KernelContext_GetInputCount = [](const OrtKernelContext *context,
                                       size_t *out) noexcept -> OrtStatus * {
    return Boundary(HostApi(), [&]() -> OrtStatus * {
      Require(context != nullptr && out != nullptr, "Null argument-count query.");
      *out = 1;
      return nullptr;
    });
  };
  api.KernelContext_GetOutputCount = api.KernelContext_GetInputCount;
  api.KernelContext_GetInput = [](const OrtKernelContext *context, size_t index,
                                  const OrtValue **out) noexcept -> OrtStatus * {
    return Boundary(HostApi(), [&]() -> OrtStatus * {
      Require(context != nullptr && out != nullptr && index == 0, "Invalid input index.");
      *out = &context->input;
      return nullptr;
    });
  };
  api.KernelContext_GetOutput = [](OrtKernelContext *context, size_t index, const int64_t *dims,
                                   size_t rank, OrtValue **out) noexcept -> OrtStatus * {
    return Boundary(HostApi(), [&]() -> OrtStatus * {
      Require(context != nullptr && out != nullptr && index == 0, "Invalid output index.");
      const Shape shape = ReadShape(dims, rank);
      const size_t bytes = TensorBytes(shape);
      if (context->tensor) {
        Require(context->tensor->shape == shape, "Custom op changed an already allocated output.");
      } else {
        context->tensor.emplace(
            Tensor::FromFloat("", shape, std::vector<float>(bytes / sizeof(float))));
        context->output.tensor = &*context->tensor;
      }
      *out = &context->output;
      return nullptr;
    });
  };
  api.GetTensorTypeAndShape = [](const OrtValue *value,
                                 OrtTensorTypeAndShapeInfo **out) noexcept -> OrtStatus * {
    return Boundary(HostApi(), [&]() -> OrtStatus * {
      Require(value != nullptr && value->tensor != nullptr && out != nullptr,
              "Null tensor shape argument.");
      *out = new OrtTensorTypeAndShapeInfo{value->tensor->shape};
      return nullptr;
    });
  };
  api.ReleaseTensorTypeAndShapeInfo = [](OrtTensorTypeAndShapeInfo *info) noexcept { delete info; };
  api.GetTensorElementType = [](const OrtTensorTypeAndShapeInfo *info,
                                ONNXTensorElementDataType *out) noexcept -> OrtStatus * {
    return Boundary(HostApi(), [&]() -> OrtStatus * {
      Require(info != nullptr && out != nullptr, "Null tensor type query.");
      *out = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      return nullptr;
    });
  };
  api.GetDimensionsCount = [](const OrtTensorTypeAndShapeInfo *info,
                              size_t *out) noexcept -> OrtStatus * {
    return Boundary(HostApi(), [&]() -> OrtStatus * {
      Require(info != nullptr && out != nullptr, "Null tensor rank query.");
      *out = info->shape.size();
      return nullptr;
    });
  };
  api.GetDimensions = [](const OrtTensorTypeAndShapeInfo *info, int64_t *out,
                         size_t count) noexcept -> OrtStatus * {
    return Boundary(HostApi(), [&]() -> OrtStatus * {
      Require(info != nullptr, "Null dimension query.");
      Require(count >= info->shape.size() && (info->shape.empty() || out != nullptr),
              "Insufficient dimension buffer.");
      if (!info->shape.empty())
        std::copy(info->shape.begin(), info->shape.end(), out);
      return nullptr;
    });
  };
  api.GetTensorShapeElementCount = [](const OrtTensorTypeAndShapeInfo *info,
                                      size_t *out) noexcept -> OrtStatus * {
    return Boundary(HostApi(), [&]() -> OrtStatus * {
      Require(info != nullptr && out != nullptr, "Null element-count argument.");
      *out = static_cast<size_t>(info->shape.product(0, info->shape.size(), "custom-op tensor"));
      return nullptr;
    });
  };
  api.GetTensorMutableData = [](OrtValue *value, void **out) noexcept -> OrtStatus * {
    return Boundary(HostApi(), [&]() -> OrtStatus * {
      Require(value != nullptr && value->tensor != nullptr && out != nullptr,
              "Null tensor data argument.");
      *out = const_cast<uint8_t *>(value->tensor->bytes());
      return nullptr;
    });
  };
  api.GetTensorMemoryInfo = [](const OrtValue *value,
                               const OrtMemoryInfo **out) noexcept -> OrtStatus * {
    return Boundary(HostApi(), [&]() -> OrtStatus * {
      Require(value != nullptr && value->tensor != nullptr && out != nullptr,
              "Null tensor memory query.");
      static const OrtMemoryInfo cpu;
      *out = &cpu;
      return nullptr;
    });
  };
  api.MemoryInfoGetDeviceType = [](const OrtMemoryInfo *, OrtMemoryInfoDeviceType *out) noexcept {
    *out = OrtMemoryInfoDeviceType_CPU;
  };
  return api;
}

} // namespace

const OrtApi &HostApi() {
  static const OrtApi api = MakeApi();
  return api;
}

const OrtApiBase &HostApiBase() {
  static const OrtApiBase base{
      [](uint32_t version) noexcept -> const OrtApi * {
        return version == 18 ? &HostApi() : nullptr;
      },
      []() noexcept { return "onnx-light-ort CPU custom-op host, API 18 subset"; }};
  return base;
}

void Check(OrtStatus *status) {
  if (status != nullptr) {
    const std::string message = status->message;
    delete status;
    throw std::runtime_error(message);
  }
}

size_t TensorBytes(const Shape &shape) {
  const int64_t count = shape.product(0, shape.size(), "custom-op tensor");
  const size_t bytes = onnx_light::core::runtime::PackedByteSize(1, count);
  Require(bytes <= 64 * 1024 * 1024, "The initial host limits each tensor to 64 MiB.");
  return bytes;
}

} // namespace onnx_light_ort::common_ort_to_light
