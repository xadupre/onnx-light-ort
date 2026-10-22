// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_proto/onnx.h"
#include "onnxruntime_c_api.h"

#include <stdexcept>
#include <vector>

namespace {

void Require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void TestModelRoundTrip() {
  onnx_light::ModelProto model;
  model.set_ir_version(10);
  model.ref_graph().set_name("dependency-contract");
  auto &node = *model.ref_graph().add_node();
  node.set_domain("onnx_light_ort.test");
  node.set_op_type("Scale");
  node.add_input("X");
  node.add_output("Y");

  onnx_light::ModelProto parsed;
  Require(parsed.ParseFromString(model.SerializeAsString()), "Model parsing failed.");
  Require(parsed.graph().name() == "dependency-contract", "Graph name did not round-trip.");
  Require(parsed.graph().node_size() == 1, "Node count did not round-trip.");
  Require(parsed.graph().node(0).domain() == "onnx_light_ort.test",
          "Custom domain did not round-trip.");
  Require(parsed.graph().node(0).input(0) == "X", "Input name did not round-trip.");
  Require(parsed.graph().node(0).output(0) == "Y", "Output name did not round-trip.");
}

void TestCustomKernel() {
  using onnx_light::core::runtime::RuntimeContext;
  using onnx_light::core::runtime::Tensor;

  onnx_light::NodeProto node;
  node.set_domain("onnx_light_ort.test");
  node.set_op_type("Scale");
  node.add_input("X");
  node.add_output("Y");

  RuntimeContext context;
  context.RegisterCustomKernel(
      node.domain(), node.op_type(), [](const onnx_light::NodeProto &op, RuntimeContext &runtime) {
        const Tensor &input = runtime.Get(op.input(0));
        std::vector<float> output(static_cast<size_t>(input.element_count()));
        for (size_t i = 0; i < output.size(); ++i)
          output[i] = 2.0f * input.AsFloat()[i];
        runtime.Put(op.output(0), Tensor::FromFloat(op.output(0), input.shape, output));
      });

  struct Case {
    std::vector<int64_t> shape;
    std::vector<float> values;
  };
  for (const auto &test : std::vector<Case>{
           {{3}, {1.0f, -2.0f, 0.5f}}, {{1, 2}, {3.0f, 4.0f}}, {{}, {-1.0f}}, {{0}, {}}}) {
    context.Put("X", Tensor::FromFloat("X", test.shape, test.values));
    onnx_light::core::runtime::RunNode(node, context);
    const Tensor &output = context.Get("Y");
    Require(output.shape == test.shape, "Output shape differs from the current input.");
    Require(output.element_count() == static_cast<int64_t>(test.values.size()),
            "Output element count differs from the current input.");
    for (size_t i = 0; i < test.values.size(); ++i) {
      Require(output.AsFloat()[i] == 2.0f * test.values[i], "Custom kernel result is incorrect.");
      Require(context.Get("X").AsFloat()[i] == test.values[i], "Custom kernel modified its input.");
    }
  }
}

} // namespace

int main() {
  TestModelRoundTrip();
  TestCustomKernel();
}
