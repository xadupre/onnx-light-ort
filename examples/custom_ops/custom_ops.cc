// SPDX-License-Identifier: Apache-2.0

#include "onnxruntime_c_api.h"

#include <array>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

class RuntimeLibrary {
public:
  explicit RuntimeLibrary(const std::filesystem::path &path) {
#ifdef _WIN32
    handle_ = LoadLibraryW(path.c_str());
    if (handle_ == nullptr)
      throw std::runtime_error("Cannot load ORT: Windows error " + std::to_string(GetLastError()));
#else
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr)
      throw std::runtime_error(dlerror());
#endif
  }
  RuntimeLibrary(const RuntimeLibrary &) = delete;
  RuntimeLibrary &operator=(const RuntimeLibrary &) = delete;
  ~RuntimeLibrary() {
#ifdef _WIN32
    FreeLibrary(handle_);
#else
    dlclose(handle_);
#endif
  }

  const OrtApi &Api() const {
    using GetApiBase = const OrtApiBase *(ORT_API_CALL *)();
#ifdef _WIN32
    const auto get_api_base =
        reinterpret_cast<GetApiBase>(GetProcAddress(handle_, "OrtGetApiBase"));
#else
    const auto get_api_base = reinterpret_cast<GetApiBase>(dlsym(handle_, "OrtGetApiBase"));
#endif
    if (get_api_base == nullptr)
      throw std::runtime_error("The runtime library does not export OrtGetApiBase.");
    const OrtApi *api = get_api_base()->GetApi(18);
    if (api == nullptr)
      throw std::runtime_error("The runtime does not support ORT API 18.");
    return *api;
  }

private:
#ifdef _WIN32
  HMODULE handle_ = nullptr;
#else
  void *handle_ = nullptr;
#endif
};

void Check(const OrtApi &api, OrtStatus *status) {
  if (status != nullptr) {
    const std::string message = api.GetErrorMessage(status);
    api.ReleaseStatus(status);
    throw std::runtime_error(message);
  }
}

template <typename T> using Owner = std::unique_ptr<T, void(ORT_API_CALL *)(T *)>;

} // namespace

int main(int argc, char **argv) {
  if (argc != 4)
    throw std::invalid_argument("Usage: custom_ops_cpp ORT_LIBRARY CUSTOM_OPS_LIBRARY MODEL");
  const RuntimeLibrary runtime(std::filesystem::absolute(argv[1]));
  const OrtApi &api = runtime.Api();
  OrtEnv *raw_env = nullptr;
  Check(api, api.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "custom-ops-example", &raw_env));
  const Owner<OrtEnv> env(raw_env, api.ReleaseEnv);
  OrtSessionOptions *raw_options = nullptr;
  Check(api, api.CreateSessionOptions(&raw_options));
  const Owner<OrtSessionOptions> options(raw_options, api.ReleaseSessionOptions);
  Check(api, api.SetIntraOpNumThreads(options.get(), 1));
  Check(api, api.SetInterOpNumThreads(options.get(), 1));
  const auto plugin = std::filesystem::absolute(argv[2]);
  Check(api, api.RegisterCustomOpsLibrary_V2(options.get(), plugin.c_str()));
  OrtSession *raw_session = nullptr;
  const auto model = std::filesystem::absolute(argv[3]);
  Check(api, api.CreateSession(env.get(), model.c_str(), options.get(), &raw_session));
  const Owner<OrtSession> session(raw_session, api.ReleaseSession);
  OrtMemoryInfo *raw_memory = nullptr;
  Check(api, api.CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &raw_memory));
  const Owner<OrtMemoryInfo> memory(raw_memory, api.ReleaseMemoryInfo);
  std::array<float, 4> values{-4, -1, 0, 2};
  const std::array<int64_t, 1> shape{4};
  OrtValue *raw_input = nullptr;
  Check(api, api.CreateTensorWithDataAsOrtValue(memory.get(), values.data(), sizeof(values),
                                                shape.data(), shape.size(),
                                                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &raw_input));
  const Owner<OrtValue> input(raw_input, api.ReleaseValue);
  const OrtValue *inputs[]{input.get()};
  const char *input_names[]{"X"};
  const char *output_names[]{"Y"};
  OrtValue *raw_output = nullptr;
  Check(api, api.Run(session.get(), nullptr, input_names, inputs, 1, output_names, 1, &raw_output));
  const Owner<OrtValue> output(raw_output, api.ReleaseValue);
  OrtTensorTypeAndShapeInfo *raw_info = nullptr;
  Check(api, api.GetTensorTypeAndShape(output.get(), &raw_info));
  const Owner<OrtTensorTypeAndShapeInfo> info(raw_info, api.ReleaseTensorTypeAndShapeInfo);
  size_t rank = 0;
  Check(api, api.GetDimensionsCount(info.get(), &rank));
  if (rank != 1)
    throw std::runtime_error("The custom op did not return a vector.");
  int64_t extent = 0;
  Check(api, api.GetDimensions(info.get(), &extent, 1));
  ONNXTensorElementDataType type;
  Check(api, api.GetTensorElementType(info.get(), &type));
  if (extent != static_cast<int64_t>(values.size()) || type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw std::runtime_error("The custom op returned an unexpected shape or type.");
  void *data = nullptr;
  Check(api, api.GetTensorMutableData(output.get(), &data));
  const auto *result = static_cast<const float *>(data);
  const std::array<float, 4> expected{-0.5f, -0.125f, 0, 2};
  for (size_t i = 0; i < expected.size(); ++i) {
    if (result[i] != expected[i])
      throw std::runtime_error("The custom op returned an unexpected value.");
    std::cout << result[i] << (i + 1 == expected.size() ? '\n' : ' ');
  }
}
