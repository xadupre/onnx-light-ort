// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_ort/common_light_to_ort/leaky_relu_kernel.h"

#include <array>
#include <cstdint>
#include <memory>

namespace onnx_light_ort::common_light_to_ort {

namespace {

#define RETURN_IF_ERROR(expression)                                                                \
  do {                                                                                             \
    if (OrtStatus *status = (expression))                                                          \
      return status;                                                                               \
  } while (false)

struct ShapeInfoDeleter {
  const OrtApi &api;
  void operator()(OrtTensorTypeAndShapeInfo *value) const noexcept {
    api.ReleaseTensorTypeAndShapeInfo(value);
  }
};

} // namespace

OrtStatus *LeakyReluKernel::Compute(OrtKernelContext *context) const {
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

} // namespace onnx_light_ort::common_light_to_ort
