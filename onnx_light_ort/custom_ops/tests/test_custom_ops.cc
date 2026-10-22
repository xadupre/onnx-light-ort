// SPDX-License-Identifier: Apache-2.0

#include "tests/ort_test_host.h"

#include <array>
#include <cmath>
#include <memory>

extern "C" OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *, const OrtApiBase *);

int main() {
  using namespace onnx_light_ort::test;
  OrtSessionOptions first;
  Check(RegisterCustomOps(&first, &api_base));
  Require(first.domains.size() == 2 && first.domains[0]->ops.size() == 1 &&
              first.domains[1]->ops.size() == 1,
          "Registration did not publish the operator.");
  const OrtCustomOp *op = first.domains[0]->ops[0];
  OrtSessionOptions second;
  Check(RegisterCustomOps(&second, &api_base));
  Require(second.domains == first.domains, "Registration did not retain stable domains.");

  OrtSessionOptions rejected;
  rejected.reject_registration = true;
  RequireError(RegisterCustomOps(&rejected, &api_base), ORT_FAIL);
  Require(rejected.domains.empty(), "Failed registration published a domain.");
  OrtSessionOptions recovered;
  Check(RegisterCustomOps(&recovered, &api_base));
  Require(recovered.domains[0] == first.domains[0], "Registration recovery lost the domain.");

  const OrtApiBase old_host{
      [](uint32_t version) noexcept -> const OrtApi * { return version == 1 ? &api : nullptr; },
      api_base.GetVersionString};
  RequireError(RegisterCustomOps(&rejected, &old_host), ORT_NOT_IMPLEMENTED);

  for (const auto &info : {OrtKernelInfo{}, OrtKernelInfo{0.25f, true}}) {
    int sentinel = 0;
    void *state = &sentinel;
    RequireError(op->CreateKernelV2(op, &api, &info, &state), ORT_INVALID_ARGUMENT);
    Require(state == nullptr, "Failed construction left a kernel pointer.");
  }

  const OrtKernelInfo info{0.25f};
  void *state = nullptr;
  Check(op->CreateKernelV2(op, &api, &info, &state));
  Require(state != nullptr, "Successful construction did not publish its kernel.");
  const std::unique_ptr<void, decltype(op->KernelDestroy)> kernel(state, op->KernelDestroy);
  std::array<float, 2> input_data{-4, 2};
  std::array<float, 2> output_data{};
  OrtValue input{{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {2}}, {}, input_data.data()};
  OrtValue output{{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {2}}, {}, output_data.data()};
  OrtKernelContext context{&input, &output};
  Check(op->KernelComputeV2(kernel.get(), &context));
  Require(output_data == std::array<float, 2>{-1, 2}, "Callback did not invoke the native kernel.");

  input.info.dimensions = {-1};
  RequireError(op->KernelComputeV2(kernel.get(), &context), ORT_INVALID_ARGUMENT);
  Require(live_shape_infos == 0, "An exception leaked input metadata across the ABI boundary.");
  input.info.dimensions = {2};
  Check(op->KernelComputeV2(kernel.get(), &context));

  const OrtCustomOp *log_op = first.domains[1]->ops[0];
  Require(std::string(log_op->GetName(log_op)) == "Log", "CPU Log was not registered.");
  void *log_state = nullptr;
  const OrtKernelInfo no_attributes;
  Check(log_op->CreateKernelV2(log_op, &api, &no_attributes, &log_state));
  const std::unique_ptr<void, decltype(log_op->KernelDestroy)> log_kernel(log_state,
                                                                          log_op->KernelDestroy);
  input_data = {0.125f, 2};
  Check(log_op->KernelComputeV2(log_kernel.get(), &context));
  for (size_t i = 0; i < input_data.size(); ++i)
    Require(std::abs(output_data[i] - std::log(input_data[i])) < 1e-5f,
            "CPU Log did not compute the logarithm.");
}
