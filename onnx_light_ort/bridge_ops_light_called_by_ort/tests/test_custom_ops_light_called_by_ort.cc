// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_ort/bridge_ops_light_called_by_ort/leaky_relu.h"
#include "tests/ort_test_host.h"

#include <array>
#include <memory>

int main() {
  using namespace onnx_light_ort::bridge_ops_light_called_by_ort;
  using namespace onnx_light_ort::test;
  std::array<float, 2> input_data{-4, 2};
  std::array<float, 2> output_data{};
  OrtValue input{{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {2}}, {}, input_data.data()};
  OrtValue output{{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {2}}, {}, output_data.data()};
  OrtKernelContext context{&input, &output};
  const auto create = [](float alpha) {
    const OrtKernelInfo info{alpha};
    void *state = nullptr;
    Check(CreateLeakyRelu(nullptr, &api, &info, &state));
    return std::unique_ptr<void, decltype(&DestroyLeakyRelu)>(state, DestroyLeakyRelu);
  };
  const auto first = create(0.0f);
  const auto second = create(0.5f);
  Check(ComputeLeakyRelu(first.get(), &context));
  Require(output_data == std::array<float, 2>{0, 2}, "Explicit zero alpha was not preserved.");
  Check(ComputeLeakyRelu(second.get(), &context));
  Require(output_data == std::array<float, 2>{-2, 2}, "Kernel attributes were not independent.");
  input.info.dimensions = {-1};
  RequireError(ComputeLeakyRelu(first.get(), &context), ORT_INVALID_ARGUMENT);
  input.info.dimensions = {2};
  Check(ComputeLeakyRelu(second.get(), &context));
  Require(output_data == std::array<float, 2>{-2, 2},
          "One kernel failure changed another instance.");
}
