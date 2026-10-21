add_library(onnx_light_ort_bridge_light_to_ort STATIC
  leaky_relu_kernel.cc
  "${onnx_light_SOURCE_DIR}/onnx_light/onnx_extensions/kernels/kernels/math/kernel_leakyrelu.cc"
)
add_library(onnx_light_ort::bridge_light_to_ort ALIAS onnx_light_ort_bridge_light_to_ort)
target_include_directories(onnx_light_ort_bridge_light_to_ort PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(onnx_light_ort_bridge_light_to_ort PUBLIC lib_onnx_core onnx_light_ort_api)
set_target_properties(onnx_light_ort_bridge_light_to_ort PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
if(MSVC)
  target_compile_options(onnx_light_ort_bridge_light_to_ort PRIVATE /W4 /WX)
else()
  target_compile_options(onnx_light_ort_bridge_light_to_ort PRIVATE -Wall -Wextra -Werror)
endif()
