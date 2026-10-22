// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_ort/bridge_ops_ort_called_by_light/host.h"
#include "onnx_proto/onnx.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void Check(OrtStatus *status) {
  if (status) {
    const std::string message = LightOrtGetErrorMessage(status);
    LightOrtReleaseStatus(status);
    throw std::runtime_error(message);
  }
}

void Require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

} // namespace

int main(int argc, char **argv) {
  Require(argc == 3, "Expected custom-op and unsupported-API library paths.");
  onnx_light::ModelProto model;
  model.set_ir_version(9);
  auto &import = *model.add_opset_import();
  import.set_domain("ort_custom_ops");
  import.set_version(1);
  auto &graph = model.ref_graph();
  auto &node = *graph.add_node();
  node.set_domain("ort_custom_ops");
  node.set_op_type("LeakyRelu");
  node.add_input("X");
  node.add_output("Y");
  auto &attribute = *node.add_attribute();
  attribute.set_name("alpha");
  attribute.set_type(onnx_light::AttributeProto::FLOAT);
  attribute.set_f(0.25f);
  for (const auto *name : {"X", "Y"}) {
    auto &value = name[0] == 'X' ? *graph.add_input() : *graph.add_output();
    value.set_name(name);
    value.ref_type().ref_tensor_type().set_elem_type(1);
  }
  const auto serialized = model.SerializeAsString();
  LightOrtSession *session = nullptr;
  Check(LightOrtCreateSession(argv[1], serialized.data(), serialized.size(), &session));
  const float *output = nullptr;
  const int64_t *shape = nullptr;
  size_t rank = 0, count = 0;
  const std::array<float, 3> input{1, -2, 4};
  const int64_t dimension = 3;
  for (int repeat = 0; repeat < 3; ++repeat) {
    Check(LightOrtRun(session, input.data(), input.size(), &dimension, 1, &output, &shape, &rank,
                      &count));
    Require(rank == 1 && shape[0] == 3 && count == 3, "Incorrect output shape.");
    Require(output[0] == 1 && output[1] == -0.5f && output[2] == 4, "Incorrect LeakyRelu output.");
  }
  Check(LightOrtRun(session, input.data(), 1, nullptr, 0, &output, &shape, &rank, &count));
  Require(rank == 0 && count == 1 && output[0] == 1, "Incorrect scalar output.");
  const int64_t zero = 0;
  Check(LightOrtRun(session, nullptr, 0, &zero, 1, &output, &shape, &rank, &count));
  Require(rank == 1 && shape[0] == 0 && count == 0, "Incorrect empty output.");
  const int64_t negative = -1;
  OrtStatus *error =
      LightOrtRun(session, input.data(), 3, &negative, 1, &output, &shape, &rank, &count);
  Require(error != nullptr, "Negative dimensions must fail.");
  LightOrtReleaseStatus(error);
  const std::array<int64_t, 2> overflow{std::numeric_limits<int64_t>::max(), 2};
  error = LightOrtRun(session, input.data(), 3, overflow.data(), 2, &output, &shape, &rank, &count);
  Require(error != nullptr, "Dimension-product overflow must fail.");
  LightOrtReleaseStatus(error);
  const int64_t too_large = 64 * 1024 * 1024;
  error = LightOrtRun(session, input.data(), static_cast<size_t>(too_large), &too_large, 1, &output,
                      &shape, &rank, &count);
  Require(error != nullptr, "Oversized tensor must fail before reading its buffer.");
  LightOrtReleaseStatus(error);
  error = LightOrtRun(session, input.data(), 2, &dimension, 1, &output, &shape, &rank, &count);
  Require(error != nullptr, "Insufficient input extent must fail.");
  LightOrtReleaseStatus(error);
  LightOrtReleaseSession(session);
  session = nullptr;
  error = LightOrtCreateSession(argv[2], serialized.data(), serialized.size(), &session);
  Require(error != nullptr && session == nullptr, "Unsupported API must fail registration.");
  LightOrtReleaseStatus(error);
}
