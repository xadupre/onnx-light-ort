// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_ort/bridge_ops_ort_called_by_light/host.h"
#include "onnx_light_ort/api_status.h"
#include "onnx_light_ort/common_ort_to_light/host_api.h"

#include "onnx_core/runtime/runtime_session.h"

#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

using namespace onnx_light::core::runtime;
using onnx_light_ort::common_ort_to_light::Check;
using onnx_light_ort::common_ort_to_light::HostApi;
using onnx_light_ort::common_ort_to_light::HostApiBase;

void Require(bool condition, const char *message) {
  if (!condition)
    throw std::invalid_argument(message);
}

class Library {
public:
  explicit Library(const char *path) {
    Require(path != nullptr, "Null plugin path.");
    const auto requested = std::filesystem::path(std::u8string(path, path + std::strlen(path)));
    Require(requested.is_absolute(), "Custom-op libraries require an explicit absolute path.");
    const auto resolved = std::filesystem::canonical(requested);
#ifdef _WIN32
    handle_ = LoadLibraryExW(resolved.c_str(), nullptr,
                             LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
    handle_ = dlopen(resolved.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    if (handle_ == nullptr)
      throw std::runtime_error("Cannot load custom-op library: " + resolved.string());
  }
  ~Library() {
#ifdef _WIN32
    FreeLibrary(handle_);
#else
    dlclose(handle_);
#endif
  }
  Library(const Library &) = delete;
  Library &operator=(const Library &) = delete;

  void Register(OrtSessionOptions &options) const {
    using RegisterFn = OrtStatus *(ORT_API_CALL *)(OrtSessionOptions *, const OrtApiBase *);
#ifdef _WIN32
    const auto function =
        reinterpret_cast<RegisterFn>(GetProcAddress(handle_, "RegisterCustomOps"));
#else
    const auto function = reinterpret_cast<RegisterFn>(dlsym(handle_, "RegisterCustomOps"));
#endif
    Require(function != nullptr, "Custom-op library has no RegisterCustomOps export.");
    Check(function(&options, &HostApiBase()));
  }

private:
#ifdef _WIN32
  HMODULE handle_ = nullptr;
#else
  void *handle_ = nullptr;
#endif
};

void ValidateDescriptor(const OrtCustomOp *op) {
  Require(op != nullptr && op->version == 18, "Custom-op host accepts only API 18 descriptors.");
  Require(op->GetName && op->GetExecutionProviderType && op->GetInputTypeCount &&
              op->GetOutputTypeCount && op->GetInputType && op->GetOutputType &&
              op->GetInputCharacteristic && op->GetOutputCharacteristic && op->GetInputMemoryType &&
              op->CreateKernelV2 && op->KernelComputeV2 && op->KernelDestroy &&
              op->GetStartVersion && op->GetEndVersion,
          "Incomplete custom-op descriptor: the host requires V2 callbacks.");
  const char *provider = op->GetExecutionProviderType(op);
  Require(provider == nullptr || std::string(provider) == "CPUExecutionProvider",
          "Only CPU custom ops are supported.");
  Require(op->GetName(op) != nullptr && *op->GetName(op) != '\0', "Empty custom-op name.");
  Require(op->GetInputTypeCount(op) == 1 && op->GetOutputTypeCount(op) == 1,
          "The initial custom-op profile requires one input and one output.");
  Require(op->GetInputType(op, 0) == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
              op->GetOutputType(op, 0) == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
          "The initial custom-op profile supports only float32.");
  Require(op->GetInputCharacteristic(op, 0) == INPUT_OUTPUT_REQUIRED &&
              op->GetOutputCharacteristic(op, 0) == INPUT_OUTPUT_REQUIRED,
          "Optional and variadic custom-op arguments are not supported.");
  const auto memory = op->GetInputMemoryType(op, 0);
  Require(memory == OrtMemTypeDefault || memory == OrtMemTypeCPUInput,
          "Custom ops must accept CPU input memory.");
  Require(op->InferOutputShapeFn == nullptr && op->GetAliasMap == nullptr &&
              op->GetMayInplace == nullptr,
          "Shape-inference and alias callbacks are not supported by the initial profile.");
}

class PluginKernel final : public KernelBase {
public:
  PluginKernel(const OrtCustomOp &op, const onnx_light::NodeProto &node,
               const KernelContext &context)
      : KernelBase(context), op_(op) {
    set_node(node);
    const OrtKernelInfo info{&node};
    OrtStatus *status = op_.CreateKernelV2(&op_, &HostApi(), &info, &state_);
    if (status != nullptr) {
      if (state_ != nullptr)
        op_.KernelDestroy(state_);
      Check(status);
    }
    Require(state_ != nullptr, "Custom-op construction returned a null kernel.");
  }
  ~PluginKernel() override { op_.KernelDestroy(state_); }

  void Run(RuntimeContext &runtime) override {
    OrtKernelContext context{};
    context.input.tensor = &runtime.Get(node_->input(0));
    Check(op_.KernelComputeV2(state_, &context));
    Require(context.tensor.has_value(), "Custom op did not produce its required output.");
    context.tensor->name = node_->output(0);
    runtime.Put(node_->output(0), std::move(*context.tensor));
  }

private:
  const OrtCustomOp &op_;
  void *state_ = nullptr;
};

} // namespace

struct LightOrtSession {
  Library library;
  onnx_light::ModelProto model;
  OrtSessionOptions registration;
  RuntimeContext context;
  std::unique_ptr<RuntimeSession> runtime;

  LightOrtSession(const char *path, const void *data, size_t size) : library(path) {
    Require(data != nullptr && size != 0 && size <= 16 * 1024 * 1024,
            "A nonempty model of at most 16 MiB is required.");
    Require(model.ParseFromString(std::string(static_cast<const char *>(data), size)),
            "Cannot parse ONNX model.");
    Require(model.has_graph() && model.graph().node_size() == 1 &&
                model.graph().input_size() == 1 && model.graph().output_size() == 1 &&
                model.graph().initializer_size() == 0 && model.functions_size() == 0,
            "The initial host supports single-node models without initializers or functions.");
    const auto &node = model.graph().node(0);
    Require(node.input_size() == 1 && node.output_size() == 1 &&
                node.input(0) == model.graph().input(0).name() &&
                node.output(0) == model.graph().output(0).name() && !node.input(0).empty() &&
                !node.output(0).empty() && node.input(0) != node.output(0),
            "The custom node must connect the distinct graph input and output.");
    for (const auto *value : {&model.graph().input(0), &model.graph().output(0)}) {
      Require(value->has_type() && value->type().has_tensor_type() &&
                  value->type().tensor_type().elem_type() == 1,
              "The host requires float32 tensor model inputs and outputs.");
    }
    int64_t version = 0;
    for (const auto &import : model.opset_import()) {
      if (import.domain() == node.domain()) {
        Require(version == 0 && import.version() > 0, "Duplicate or invalid custom opset.");
        version = import.version();
      }
    }
    Require(version > 0 && !node.domain().empty(), "A custom-domain opset import is required.");
    library.Register(registration);
    std::set<std::pair<std::string, std::string>> names;
    const OrtCustomOp *selected = nullptr;
    for (const auto *domain : registration.domains) {
      Require(domain != nullptr, "Null registered custom-op domain.");
      for (const auto *op : domain->ops) {
        ValidateDescriptor(op);
        Require(names.emplace(domain->name, op->GetName(op)).second,
                "Duplicate custom-op registrations are not supported.");
        const int start = op->GetStartVersion(op);
        const int end = op->GetEndVersion(op);
        Require(start > 0 && end >= start, "Invalid custom-op version range.");
        if (domain->name == node.domain() && op->GetName(op) == node.op_type() &&
            version >= start && version <= end)
          selected = op;
      }
    }
    Require(selected != nullptr, "No compatible custom-op domain, name, or opset.");
    context.RegisterKernelFn(
        node.domain(), node.op_type(), onnx_light::core::symbolic::Device::kCPU,
        [selected](const onnx_light::NodeProto &op, RuntimeContext &, const KernelContext &ctx) {
          return std::make_unique<PluginKernel>(*selected, op, ctx);
        });
    RuntimeSessionOptions options;
    options.allow_external_output_allocators = true;
    options.check_shapes = true;
    runtime = std::make_unique<RuntimeSession>(model, options);
  }
};

extern "C" {

OrtStatus *LightOrtCreateSession(const char *library, const void *model, size_t model_size,
                                 LightOrtSession **out) {
  return onnx_light_ort::Boundary(HostApi(), [&]() -> OrtStatus * {
    Require(out != nullptr, "Null session output.");
    *out = nullptr;
    auto session = std::make_unique<LightOrtSession>(library, model, model_size);
    *out = session.release();
    return nullptr;
  });
}

OrtStatus *LightOrtRun(LightOrtSession *session, const float *input, size_t input_count,
                       const int64_t *dimensions, size_t rank, const float **output,
                       const int64_t **output_shape, size_t *output_rank, size_t *output_count) {
  return onnx_light_ort::Boundary(HostApi(), [&]() -> OrtStatus * {
    Require(session && output && output_shape && output_rank && output_count,
            "Null custom-op run argument.");
    Require(rank <= Shape::kMaxRank && (rank == 0 || dimensions != nullptr),
            "Custom-op tensors support at most 16 dimensions.");
    Shape shape;
    for (size_t i = 0; i < rank; ++i)
      shape.push_back(dimensions[i]);
    const int64_t count = shape.product(0, rank, "custom-op input");
    Require(static_cast<uint64_t>(count) == input_count, "Input buffer size differs from shape.");
    const size_t bytes = onnx_light_ort::common_ort_to_light::TensorBytes(shape);
    Require(bytes == 0 ||
                (input != nullptr && reinterpret_cast<uintptr_t>(input) % alignof(float) == 0),
            "Custom-op input must have float-aligned storage.");
    const auto &graph = session->model.graph();
    session->context.Put(graph.input(0).name(),
                         Tensor::Borrow(graph.input(0).name(), 1, shape,
                                        reinterpret_cast<const uint8_t *>(input), bytes));
    struct InputLifetime {
      RuntimeContext &context;
      std::string name;
      ~InputLifetime() { context.tensors().erase(name); }
    } input_lifetime{session->context, graph.input(0).name()};
    session->runtime->Run(session->context);
    const Tensor &result = session->context.Get(graph.output(0).name());
    *output = result.AsFloat();
    *output_shape = result.shape.data();
    *output_rank = result.shape.size();
    *output_count = static_cast<size_t>(result.element_count());
    return nullptr;
  });
}

void LightOrtReleaseSession(LightOrtSession *session) { delete session; }
const char *LightOrtGetErrorMessage(const OrtStatus *status) { return status->message.c_str(); }
void LightOrtReleaseStatus(OrtStatus *status) { delete status; }

} // extern "C"
