cmake_minimum_required(VERSION 3.28)
project(onnx_light_ort VERSION 0.1.0 LANGUAGES CXX)

option(ONNX_LIGHT_ORT_BUILD_HOST "Expose the standalone compatibility-host target." ON)
option(ONNX_LIGHT_ORT_BUILD_PLUGIN "Expose the plugin-for-ORT target." ON)
option(ONNX_LIGHT_ORT_BUILD_TESTS "Build dependency integration tests." ON)
option(ONNX_LIGHT_ORT_BUILD_FUZZERS "Build onnx-light libFuzzer targets." OFF)
option(ONNX_LIGHT_ORT_TEST_WITH_ORT "Run integration tests in a real ORT host." OFF)
option(ONNX_LIGHT_ORT_TEST_PYTHON "Run Python tests for the standalone custom-op host." OFF)
option(ONNX_LIGHT_ORT_BUILD_EXAMPLES "Build the C++ custom-op example when plugins are enabled." ON)
set(ONNX_LIGHT_ORT_RUNTIME_LIBRARY "" CACHE FILEPATH "ORT shared library for the C++ example test.")

if(NOT ONNX_LIGHT_ORT_BUILD_HOST AND NOT ONNX_LIGHT_ORT_BUILD_PLUGIN)
  message(FATAL_ERROR "Enable at least one of the host and plugin targets.")
endif()
if(ONNX_LIGHT_ORT_TEST_WITH_ORT AND
   (NOT ONNX_LIGHT_ORT_BUILD_PLUGIN OR NOT ONNX_LIGHT_ORT_BUILD_TESTS))
  message(FATAL_ERROR "ORT integration tests require the plugin and tests targets.")
endif()
if(ONNX_LIGHT_ORT_TEST_PYTHON AND
   (NOT ONNX_LIGHT_ORT_BUILD_HOST OR NOT ONNX_LIGHT_ORT_BUILD_TESTS))
  message(FATAL_ERROR "Standalone Python tests require the host and tests targets.")
endif()
if(ONNX_LIGHT_ORT_TEST_PYTHON OR ONNX_LIGHT_ORT_TEST_WITH_ORT)
  find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
endif()

include(cmake/Dependencies.cmake)

if(ONNX_LIGHT_ORT_BUILD_TESTS)
  enable_testing()
  include(cmake/Testing.cmake)
endif()

if(ONNX_LIGHT_ORT_BUILD_HOST)
  add_subdirectory(onnx_light_ort/common_ort_to_light)
  add_subdirectory(onnx_light_ort/bridge_ops_ort_to_light)
  add_subdirectory(onnx_light_ort/bridge_ep_ort_to_light)
endif()

if(ONNX_LIGHT_ORT_BUILD_PLUGIN)
  add_subdirectory(onnx_light_ort/common_light_to_ort)
  add_subdirectory(onnx_light_ort/bridge_ops_light_to_ort)
  add_subdirectory(onnx_light_ort/bridge_ep_light_to_ort)
  add_subdirectory(onnx_light_ort/custom_ops)
endif()

if(ONNX_LIGHT_ORT_BUILD_TESTS)
  if(ONNX_LIGHT_ORT_TEST_WITH_ORT)
    add_test(NAME ort_custom_ops
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ort_custom_ops.py"
        --library "$<TARGET_FILE:onnx_light_ort_custom_ops>"
    )
    set_tests_properties(ort_custom_ops PROPERTIES TIMEOUT 120)
    set_tests_properties(ort_custom_ops PROPERTIES
      LABELS "integration;custom_ops;bridge_ops_light_to_ort;common_light_to_ort")
  endif()
endif()

if(ONNX_LIGHT_ORT_BUILD_PLUGIN AND ONNX_LIGHT_ORT_BUILD_EXAMPLES)
  add_subdirectory(examples/custom_ops)
endif()

if(ONNX_LIGHT_ORT_TEST_PYTHON AND ONNX_LIGHT_ORT_TEST_WITH_ORT AND
   ONNX_LIGHT_ORT_BUILD_EXAMPLES)
  add_test(NAME gallery_light_in_ort
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/docs/examples/plot_light_in_ort.py")
  set_tests_properties(gallery_light_in_ort PROPERTIES LABELS "example;python;custom_ops"
    TIMEOUT 60 ENVIRONMENT "ONNX_LIGHT_ORT_PLUGIN_LIBRARY=$<TARGET_FILE:onnx_light_ort_custom_ops>")
  add_test(NAME gallery_ort_in_light
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/docs/examples/plot_ort_in_light.py")
  set_tests_properties(gallery_ort_in_light PROPERTIES LABELS "example;python;bridge_ops_ort_to_light"
    TIMEOUT 60 ENVIRONMENT
    "PYTHONPATH=${PROJECT_SOURCE_DIR};ONNX_LIGHT_ORT_HOST_LIBRARY=$<TARGET_FILE:onnx_light_ort_bridge_ops_ort_to_light>;ONNX_LIGHT_ORT_FIXTURE_LIBRARY=$<TARGET_FILE:onnx_light_ort_scale_fixture>")
endif()
