add_library(onnx_light_ort_bridge_ops_ort_to_light SHARED host.cc)
add_library(onnx_light_ort::bridge_ops_ort_to_light ALIAS onnx_light_ort_bridge_ops_ort_to_light)
target_link_libraries(onnx_light_ort_bridge_ops_ort_to_light PRIVATE onnx_light_ort::common_ort_to_light
  ${CMAKE_DL_LIBS})
target_compile_definitions(onnx_light_ort_bridge_ops_ort_to_light PRIVATE LIGHT_ORT_BUILDING_HOST)
set_target_properties(onnx_light_ort_bridge_ops_ort_to_light PROPERTIES
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
if(UNIX AND NOT APPLE)
  target_link_options(onnx_light_ort_bridge_ops_ort_to_light PRIVATE "LINKER:--exclude-libs,ALL")
endif()

if(ONNX_LIGHT_ORT_BUILD_TESTS OR ONNX_LIGHT_ORT_BUILD_EXAMPLES)
  add_library(onnx_light_ort_scale_fixture SHARED
    "${PROJECT_SOURCE_DIR}/examples/custom_ops/scale_fixture.cc")
  target_link_libraries(onnx_light_ort_scale_fixture PRIVATE onnx_light_ort_api)
  set_target_properties(onnx_light_ort_scale_fixture PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
endif()

if(ONNX_LIGHT_ORT_BUILD_TESTS)
  add_library(onnx_light_ort_bad_version_fixture SHARED
    "${PROJECT_SOURCE_DIR}/examples/custom_ops/scale_fixture.cc")
  target_link_libraries(onnx_light_ort_bad_version_fixture PRIVATE onnx_light_ort_api)
  target_compile_definitions(onnx_light_ort_bad_version_fixture PRIVATE FIXTURE_BAD_VERSION)
  onnx_light_ort_add_test(onnx_light_ort_host_test tests/test_host.cc custom_op_host
    "$<TARGET_FILE:onnx_light_ort_scale_fixture>"
    "$<TARGET_FILE:onnx_light_ort_bad_version_fixture>")
  target_link_libraries(onnx_light_ort_host_test PRIVATE
    onnx_light_ort::bridge_ops_ort_to_light lib_onnx_proto)
  add_dependencies(onnx_light_ort_host_test onnx_light_ort_scale_fixture onnx_light_ort_bad_version_fixture)
  set_tests_properties(custom_op_host PROPERTIES
    LABELS "unit;bridge_ops_ort_to_light;common_ort_to_light")
  if(ONNX_LIGHT_ORT_TEST_PYTHON)
    add_test(NAME custom_op_host_python
      COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_host_custom_ops.py"
        --host "$<TARGET_FILE:onnx_light_ort_bridge_ops_ort_to_light>"
        --library "$<TARGET_FILE:onnx_light_ort_scale_fixture>"
        --bad-library "$<TARGET_FILE:onnx_light_ort_bad_version_fixture>")
    set_tests_properties(custom_op_host_python PROPERTIES
      LABELS "integration;python;bridge_ops_ort_to_light"
      TIMEOUT 120 ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}")
  endif()
endif()

foreach(target onnx_light_ort_bridge_ops_ort_to_light
    onnx_light_ort_scale_fixture onnx_light_ort_bad_version_fixture)
  if(TARGET "${target}")
    if(MSVC)
      target_compile_options("${target}" PRIVATE /W4 /WX)
    else()
      target_compile_options("${target}" PRIVATE -Wall -Wextra -Werror)
    endif()
  endif()
endforeach()
