cmake_minimum_required(VERSION 3.28)
project(onnx_light_ort VERSION 0.1.0 LANGUAGES CXX)

option(ONNX_LIGHT_ORT_BUILD_HOST "Expose the standalone compatibility-host target." ON)
option(ONNX_LIGHT_ORT_BUILD_PLUGIN "Expose the plugin-for-ORT target." ON)
option(ONNX_LIGHT_ORT_BUILD_TESTS "Build dependency integration tests." ON)
option(ONNX_LIGHT_ORT_TEST_WITH_ORT "Run integration tests in a real ORT host." OFF)

if(NOT ONNX_LIGHT_ORT_BUILD_HOST AND NOT ONNX_LIGHT_ORT_BUILD_PLUGIN)
  message(FATAL_ERROR "Enable at least one of the host and plugin targets.")
endif()
if(ONNX_LIGHT_ORT_TEST_WITH_ORT AND
   (NOT ONNX_LIGHT_ORT_BUILD_PLUGIN OR NOT ONNX_LIGHT_ORT_BUILD_TESTS))
  message(FATAL_ERROR "ORT integration tests require the plugin and tests targets.")
endif()

include(cmake/Dependencies.cmake)

if(ONNX_LIGHT_ORT_BUILD_HOST)
  add_subdirectory(onnx_light_ort/bridge_ort_to_light)
endif()

if(ONNX_LIGHT_ORT_BUILD_PLUGIN)
  add_subdirectory(onnx_light_ort/bridge_light_to_ort)
  add_subdirectory(onnx_light_ort/custom_ops)
endif()

if(ONNX_LIGHT_ORT_BUILD_TESTS)
  enable_testing()
  add_executable(onnx_light_ort_dependency_test tests/test_dependency_contract.cc)
  target_link_libraries(onnx_light_ort_dependency_test PRIVATE lib_onnx_core onnx_light_ort_api)
  target_compile_features(onnx_light_ort_dependency_test PRIVATE cxx_std_20)
  if(MSVC)
    target_compile_options(onnx_light_ort_dependency_test PRIVATE /W4 /WX)
  else()
    target_compile_options(onnx_light_ort_dependency_test PRIVATE -Wall -Wextra -Werror)
  endif()
  if(ONNX_LIGHT_ORT_BUILD_HOST)
    target_link_libraries(onnx_light_ort_dependency_test PRIVATE onnx_light_ort::bridge_ort_to_light)
  endif()
  if(ONNX_LIGHT_ORT_BUILD_PLUGIN)
    add_dependencies(onnx_light_ort_dependency_test onnx_light_ort_custom_ops)
  endif()
  add_test(NAME dependency_contract COMMAND onnx_light_ort_dependency_test)
  set_tests_properties(dependency_contract PROPERTIES TIMEOUT 60)
  if(ONNX_LIGHT_ORT_TEST_WITH_ORT)
    find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
    add_test(NAME ort_custom_ops
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ort_custom_ops.py"
        --library "$<TARGET_FILE:onnx_light_ort_custom_ops>"
    )
    set_tests_properties(ort_custom_ops PROPERTIES TIMEOUT 120)
  endif()
endif()
