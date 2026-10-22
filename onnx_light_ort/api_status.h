// SPDX-License-Identifier: Apache-2.0

#ifndef ONNX_LIGHT_ORT_API_STATUS_H
#define ONNX_LIGHT_ORT_API_STATUS_H

#include "onnxruntime_c_api.h"

#include <exception>
#include <stdexcept>

namespace onnx_light_ort {

template <typename Function> OrtStatus *Boundary(const OrtApi &api, Function &&function) noexcept {
  try {
    return function();
  } catch (const std::invalid_argument &error) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, error.what());
  } catch (const std::exception &error) {
    return api.CreateStatus(ORT_RUNTIME_EXCEPTION, error.what());
  } catch (...) {
    return api.CreateStatus(ORT_RUNTIME_EXCEPTION, "Unknown exception in onnx-light custom op.");
  }
}

} // namespace onnx_light_ort

#endif
