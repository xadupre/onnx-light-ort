// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_ort/common_light_to_ort/leaky_relu_kernel.h"
#include "tests/ort_test_host.h"

#include <array>
#include <limits>

int main() {
  using namespace onnx_light_ort::test;
  const onnx_light_ort::common_light_to_ort::LeakyReluKernel kernel{api, 0.125f};
  const std::array<float, 4> original{-4, -1, 0, 2};
  std::array<float, 4> input_data = original;
  std::array<float, 4> output_data{};
  OrtValue input{{}, {}, input_data.data()};
  OrtValue output{{}, {}, output_data.data()};
  OrtKernelContext context{&input, &output};
  struct Case {
    std::vector<int64_t> shape;
    size_t count;
  };
  for (const auto &test : std::vector<Case>{
           {{4}, 4}, {{2, 2}, 4}, {{}, 1}, {{0}, 0}, {std::vector<int64_t>(16, 1), 1}}) {
    input.info.dimensions = output.info.dimensions = test.shape;
    input.data = test.count == 0 ? nullptr : input_data.data();
    output.data = test.count == 0 ? nullptr : output_data.data();
    Check(kernel.Compute(&context));
    const std::array<float, 4> expected{-0.5f, -0.125f, 0, 2};
    for (size_t i = 0; i < test.count; ++i)
      Require(output_data[i] == expected[i], "Native result differs for the current shape.");
    Require(input_data == original, "Computation modified its input.");
    Require(live_shape_infos == 0, "Successful execution leaked metadata.");
  }

  input.info.dimensions = output.info.dimensions = {4};
  input.data = input_data.data();
  output.data = output_data.data();
  context.input = nullptr;
  RequireError(kernel.Compute(&context), ORT_INVALID_ARGUMENT);
  context.input = &input;
  context.output = nullptr;
  RequireError(kernel.Compute(&context), ORT_INVALID_ARGUMENT);
  context.output = &output;
  input.memory.device = OrtMemoryInfoDeviceType_GPU;
  RequireError(kernel.Compute(&context), ORT_INVALID_ARGUMENT);
  input.memory.device = OrtMemoryInfoDeviceType_CPU;
  input.info.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  RequireError(kernel.Compute(&context), ORT_INVALID_ARGUMENT);
  input.info.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  output.info.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  RequireError(kernel.Compute(&context), ORT_INVALID_ARGUMENT);
  output.info.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  input.info.dimensions.assign(17, 1);
  RequireError(kernel.Compute(&context), ORT_INVALID_ARGUMENT);

  for (const auto &shape : std::vector<std::vector<int64_t>>{
           {-1}, {std::numeric_limits<int64_t>::max(), 2}, {std::numeric_limits<int64_t>::max()}}) {
    input.info.dimensions = shape;
    bool failed = false;
    try {
      static_cast<void>(kernel.Compute(&context));
    } catch (const std::invalid_argument &) {
      failed = true;
    }
    Require(failed, "Invalid dimensions or byte counts were accepted.");
    Require(live_shape_infos == 0, "Validation failure leaked metadata.");
  }
  input.info.dimensions = {4};
  input.data = nullptr;
  RequireError(kernel.Compute(&context), ORT_INVALID_ARGUMENT);
  input.data = input_data.data();
  output.data = input_data.data();
  RequireError(kernel.Compute(&context), ORT_INVALID_ARGUMENT);
  output.data = output_data.data();

  OrtStatus *injected = api.CreateStatus(ORT_FAIL, "Input acquisition failed.");
  context.input_error = injected;
  Require(kernel.Compute(&context) == injected, "Host status was not propagated unchanged.");
  api.ReleaseStatus(injected);
  context.input_error = nullptr;
  Check(kernel.Compute(&context));
  Require(live_shape_infos == 0, "Failure recovery leaked metadata.");
}
