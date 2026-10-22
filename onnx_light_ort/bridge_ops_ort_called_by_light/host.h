// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnxruntime_c_api.h"

#ifdef _WIN32
#ifdef LIGHT_ORT_BUILDING_HOST
#define LIGHT_ORT_EXPORT __declspec(dllexport)
#else
#define LIGHT_ORT_EXPORT __declspec(dllimport)
#endif
#else
#define LIGHT_ORT_EXPORT __attribute__((visibility("default")))
#endif

struct LightOrtSession;

extern "C" {
LIGHT_ORT_EXPORT OrtStatus *LightOrtCreateSession(const char *library, const void *model,
                                                  size_t model_size, LightOrtSession **out);
LIGHT_ORT_EXPORT OrtStatus *LightOrtRun(LightOrtSession *session, const float *input,
                                        size_t input_count, const int64_t *shape, size_t rank,
                                        const float **output, const int64_t **output_shape,
                                        size_t *output_rank, size_t *output_count);
LIGHT_ORT_EXPORT OrtStatus *LightOrtRunTyped(LightOrtSession *session, const void *input,
                                             ONNXTensorElementDataType input_type,
                                             size_t input_count, const int64_t *shape, size_t rank,
                                             const void **output, const int64_t **output_shape,
                                             size_t *output_rank, size_t *output_count);
LIGHT_ORT_EXPORT void LightOrtReleaseSession(LightOrtSession *session);
LIGHT_ORT_EXPORT const char *LightOrtGetErrorMessage(const OrtStatus *status);
LIGHT_ORT_EXPORT void LightOrtReleaseStatus(OrtStatus *status);
}
