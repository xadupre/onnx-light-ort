add_library(onnx_light_ort_custom_ops SHARED custom_ops.cc)
add_library(onnx_light_ort::custom_ops ALIAS onnx_light_ort_custom_ops)
add_library(onnx_light_ort::plugin ALIAS onnx_light_ort_custom_ops)
target_link_libraries(onnx_light_ort_custom_ops PRIVATE onnx_light_ort::bridge_ops_light_called_by_ort)
set_target_properties(onnx_light_ort_custom_ops PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
if(APPLE)
  target_link_options(onnx_light_ort_custom_ops PRIVATE "LINKER:-exported_symbol,_RegisterCustomOps")
elseif(UNIX)
  target_link_options(onnx_light_ort_custom_ops PRIVATE "LINKER:--exclude-libs,ALL")
endif()

if(MSVC)
  target_compile_options(onnx_light_ort_custom_ops PRIVATE /W4 /WX)
else()
  target_compile_options(onnx_light_ort_custom_ops PRIVATE -Wall -Wextra -Werror)
endif()

if(ONNX_LIGHT_ORT_BUILD_TESTS)
  onnx_light_ort_add_test(onnx_light_ort_custom_ops_test tests/test_custom_ops.cc custom_ops_unit)
  target_link_libraries(onnx_light_ort_custom_ops_test PRIVATE onnx_light_ort::custom_ops)
  set_tests_properties(custom_ops_unit PROPERTIES LABELS "unit;custom_ops")
endif()
