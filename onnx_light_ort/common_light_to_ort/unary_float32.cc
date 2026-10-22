// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_ort/common_light_to_ort/unary_float32.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>

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

OrtStatus *GetUnaryFloat32Buffers(const OrtApi &api, OrtKernelContext *context, const char *name,
                                  UnaryFloat32Buffers &buffers) {
  using onnx_light::core::runtime::DataType;
  using onnx_light::core::runtime::Shape;
  const auto invalid = [&](const char *message) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, (std::string(name) + message).c_str());
  };

  const OrtValue *input = nullptr;
  RETURN_IF_ERROR(api.KernelContext_GetInput(context, 0, &input));
  if (input == nullptr)
    return invalid(" requires input X.");

  const OrtMemoryInfo *memory = nullptr;
  RETURN_IF_ERROR(api.GetTensorMemoryInfo(input, &memory));
  OrtMemoryInfoDeviceType device;
  api.MemoryInfoGetDeviceType(memory, &device);
  if (device != OrtMemoryInfoDeviceType_CPU)
    return invalid(" requires CPU memory.");

  OrtTensorTypeAndShapeInfo *raw_info = nullptr;
  RETURN_IF_ERROR(api.GetTensorTypeAndShape(input, &raw_info));
  const std::unique_ptr<OrtTensorTypeAndShapeInfo, ShapeInfoDeleter> info(raw_info, {api});
  ONNXTensorElementDataType type;
  RETURN_IF_ERROR(api.GetTensorElementType(info.get(), &type));
  if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    return invalid(" requires a float32 tensor.");
  size_t rank = 0;
  RETURN_IF_ERROR(api.GetDimensionsCount(info.get(), &rank));
  if (rank > Shape::kMaxRank)
    return invalid(" supports at most 16 dimensions.");
  std::array<int64_t, Shape::kMaxRank> dimensions{};
  RETURN_IF_ERROR(api.GetDimensions(info.get(), dimensions.data(), rank));
  Shape shape;
  for (size_t i = 0; i < rank; ++i)
    shape.push_back(dimensions[i]);
  const int64_t elements = shape.product(0, rank, "ORT unary float32 input");
  const size_t bytes =
      onnx_light::core::runtime::PackedByteSize(static_cast<int32_t>(DataType::FLOAT), elements);

  void *input_data = nullptr;
  // API 18 exposes tensor data through a mutable accessor even for read-only inputs.
  RETURN_IF_ERROR(api.GetTensorMutableData(const_cast<OrtValue *>(input), &input_data));
  OrtValue *output = nullptr;
  RETURN_IF_ERROR(api.KernelContext_GetOutput(context, 0, dimensions.data(), rank, &output));
  if (output == nullptr)
    return invalid(" requires output Y.");
  void *output_data = nullptr;
  RETURN_IF_ERROR(api.GetTensorMutableData(output, &output_data));
  if (bytes != 0 && (input_data == nullptr || output_data == nullptr))
    return invalid(" received a null tensor buffer.");
  if (bytes != 0 && (reinterpret_cast<uintptr_t>(input_data) % alignof(float) != 0 ||
                     reinterpret_cast<uintptr_t>(output_data) % alignof(float) != 0))
    return invalid(" requires float-aligned buffers.");
  const uintptr_t source = reinterpret_cast<uintptr_t>(input_data);
  const uintptr_t destination = reinterpret_cast<uintptr_t>(output_data);
  if (bytes != 0 &&
      (source <= destination ? destination - source < bytes : source - destination < bytes))
    return invalid(" requires non-overlapping buffers.");

  buffers = {shape, static_cast<const float *>(input_data), static_cast<float *>(output_data),
             bytes / sizeof(float)};
  return nullptr;
}

} // namespace onnx_light_ort::common_light_to_ort
