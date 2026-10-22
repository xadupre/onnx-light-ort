add_library(onnx_light_ort_common_light_to_ort STATIC
  leaky_relu_kernel.cc
  unary_float32.cc
  log_kernel.cc
  "${onnx_light_SOURCE_DIR}/onnx_light/onnx_extensions/kernels/kernels/math/kernel_leakyrelu.cc"
)
add_library(onnx_light_ort::common_light_to_ort ALIAS onnx_light_ort_common_light_to_ort)
target_include_directories(onnx_light_ort_common_light_to_ort PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(onnx_light_ort_common_light_to_ort PUBLIC lib_onnx_core onnx_light_ort_api)
target_link_libraries(onnx_light_ort_common_light_to_ort PRIVATE ml_cpu_compute::ml_cpu_compute)
set_target_properties(onnx_light_ort_common_light_to_ort PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
if(MSVC)
  target_compile_options(onnx_light_ort_common_light_to_ort PRIVATE /W4 /WX)
else()
  target_compile_options(onnx_light_ort_common_light_to_ort PRIVATE -Wall -Wextra -Werror)
endif()

if(ONNX_LIGHT_ORT_BUILD_TESTS)
  onnx_light_ort_add_test(onnx_light_ort_common_light_to_ort_test tests/test_leaky_relu_kernel.cc
    common_light_to_ort_unit)
  target_link_libraries(onnx_light_ort_common_light_to_ort_test PRIVATE onnx_light_ort::common_light_to_ort)
  set_tests_properties(common_light_to_ort_unit PROPERTIES LABELS "unit;common_light_to_ort")
endif()
