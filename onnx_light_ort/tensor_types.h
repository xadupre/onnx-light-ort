// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnxruntime_c_api.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace onnx_light_ort {

inline constexpr std::array<ONNXTensorElementDataType, 4> kSupportedTensorTypes{
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16};

inline bool IsSupportedTensorType(int32_t type) {
  return std::find(kSupportedTensorTypes.begin(), kSupportedTensorTypes.end(), type) !=
         kSupportedTensorTypes.end();
}

inline size_t TensorAlignment(ONNXTensorElementDataType type) {
  switch (type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return alignof(float);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return alignof(double);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    return alignof(uint16_t);
  default:
    throw std::invalid_argument(
        "Supported tensor types are float16, bfloat16, float32 and float64.");
  }
}

} // namespace onnx_light_ort
