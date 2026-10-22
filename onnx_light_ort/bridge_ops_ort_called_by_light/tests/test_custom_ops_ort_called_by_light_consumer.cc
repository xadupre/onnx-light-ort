// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_ort/bridge_ops_ort_called_by_light/host.h"

#include <stdexcept>

int main() {
  LightOrtSession *session = nullptr;
  OrtStatus *status = LightOrtCreateSession(nullptr, nullptr, 0, &session);
  if (status == nullptr)
    throw std::runtime_error("The public host API must reject an empty session request.");
  LightOrtReleaseStatus(status);
  if (session != nullptr)
    throw std::runtime_error("A failed request must not publish a session.");
}
