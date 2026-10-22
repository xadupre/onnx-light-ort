add_library(onnx_light_ort_bridge_ops_ort_called_by_light SHARED host.cc)
add_library(onnx_light_ort::bridge_ops_ort_called_by_light ALIAS onnx_light_ort_bridge_ops_ort_called_by_light)
target_link_libraries(onnx_light_ort_bridge_ops_ort_called_by_light
  PUBLIC onnx_light_ort_api
  PRIVATE onnx_light_ort::common_ort_to_light ${CMAKE_DL_LIBS})
target_include_directories(onnx_light_ort_bridge_ops_ort_called_by_light PUBLIC
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_compile_definitions(onnx_light_ort_bridge_ops_ort_called_by_light PRIVATE LIGHT_ORT_BUILDING_HOST)
set_target_properties(onnx_light_ort_bridge_ops_ort_called_by_light PROPERTIES
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
if(UNIX AND NOT APPLE)
  target_link_options(onnx_light_ort_bridge_ops_ort_called_by_light PRIVATE "LINKER:--exclude-libs,ALL")
endif()

if(ONNX_LIGHT_ORT_BUILD_TESTS OR ONNX_LIGHT_ORT_BUILD_EXAMPLES)
  add_library(onnx_light_ort_ort_custom_ops SHARED
    "${PROJECT_SOURCE_DIR}/examples/custom_ops/ort_custom_ops.cc")
  target_link_libraries(onnx_light_ort_ort_custom_ops PRIVATE onnx_light_ort_api)
  set_target_properties(onnx_light_ort_ort_custom_ops PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
endif()

if(ONNX_LIGHT_ORT_BUILD_TESTS)
  # The test helper's extra includes would hide missing public requirements.
  add_executable(onnx_light_ort_custom_ops_ort_called_by_light_consumer_test
    tests/test_custom_ops_ort_called_by_light_consumer.cc)
  target_link_libraries(onnx_light_ort_custom_ops_ort_called_by_light_consumer_test PRIVATE
    onnx_light_ort::bridge_ops_ort_called_by_light)
  add_test(NAME custom_ops_ort_called_by_light_consumer
    COMMAND onnx_light_ort_custom_ops_ort_called_by_light_consumer_test)
  set_tests_properties(custom_ops_ort_called_by_light_consumer PROPERTIES
    TIMEOUT 60 LABELS "unit;bridge_ops_ort_called_by_light")
  if(WIN32 AND ONNX_LIGHT_BUILD_PYTHON)
    set_tests_properties(custom_ops_ort_called_by_light_consumer PROPERTIES ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:${ONNX_LIGHT_ORT_PYTHON_DIR}/onnx_light/onnx_py")
  endif()
  add_library(onnx_light_ort_unsupported_api_custom_ops SHARED
    "${PROJECT_SOURCE_DIR}/examples/custom_ops/ort_custom_ops.cc")
  target_link_libraries(onnx_light_ort_unsupported_api_custom_ops PRIVATE onnx_light_ort_api)
  target_compile_definitions(onnx_light_ort_unsupported_api_custom_ops PRIVATE CUSTOM_OP_UNSUPPORTED_API)
  onnx_light_ort_add_test(onnx_light_ort_custom_ops_ort_called_by_light_test
    tests/test_custom_ops_ort_called_by_light.cc custom_ops_ort_called_by_light_unit
    "$<TARGET_FILE:onnx_light_ort_ort_custom_ops>"
    "$<TARGET_FILE:onnx_light_ort_unsupported_api_custom_ops>")
  target_link_libraries(onnx_light_ort_custom_ops_ort_called_by_light_test PRIVATE
    onnx_light_ort::bridge_ops_ort_called_by_light lib_onnx_proto)
  add_dependencies(onnx_light_ort_custom_ops_ort_called_by_light_test
    onnx_light_ort_ort_custom_ops onnx_light_ort_unsupported_api_custom_ops)
  set_tests_properties(custom_ops_ort_called_by_light_unit PROPERTIES
    LABELS "unit;bridge_ops_ort_called_by_light;common_ort_to_light")
  if(ONNX_LIGHT_ORT_TEST_PYTHON)
    add_test(NAME custom_ops_ort_called_by_light_python
      COMMAND ${onnx_light_ort_python_test_command} "${PROJECT_SOURCE_DIR}/tests/test_custom_ops_ort_called_by_light.py"
        --host "$<TARGET_FILE:onnx_light_ort_bridge_ops_ort_called_by_light>"
        --library "$<TARGET_FILE:onnx_light_ort_ort_custom_ops>"
        --bad-library "$<TARGET_FILE:onnx_light_ort_unsupported_api_custom_ops>")
    set_tests_properties(custom_ops_ort_called_by_light_python PROPERTIES
      LABELS "integration;python;bridge_ops_ort_called_by_light"
      TIMEOUT 120 ENVIRONMENT_MODIFICATION "${onnx_light_ort_python_environment}")
  endif()
endif()

foreach(target onnx_light_ort_bridge_ops_ort_called_by_light
    onnx_light_ort_custom_ops_ort_called_by_light_consumer_test
    onnx_light_ort_ort_custom_ops onnx_light_ort_unsupported_api_custom_ops)
  if(TARGET "${target}")
    if(MSVC)
      target_compile_options("${target}" PRIVATE /W4 /WX)
    else()
      target_compile_options("${target}" PRIVATE -Wall -Wextra -Werror)
    endif()
  endif()
endforeach()
