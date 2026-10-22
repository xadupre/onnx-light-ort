cmake_minimum_required(VERSION 3.28)
project(onnx_light_ort VERSION 0.1.0 LANGUAGES CXX)

option(ONNX_LIGHT_ORT_BUILD_HOST "Expose the standalone compatibility-host target." ON)
option(ONNX_LIGHT_ORT_BUILD_PLUGIN "Expose the plugin-for-ORT target." ON)
option(ONNX_LIGHT_ORT_BUILD_TESTS "Build dependency integration tests." ON)
option(ONNX_LIGHT_ORT_BUILD_FUZZERS "Build onnx-light libFuzzer targets." OFF)
option(ONNX_LIGHT_ORT_TEST_WITH_ORT "Run integration tests in a real ORT host." OFF)
option(ONNX_LIGHT_ORT_TEST_PYTHON "Run Python tests for the standalone custom-op host." OFF)
option(ONNX_LIGHT_ORT_BUILD_PYTHON "Build onnx-light Python model bindings for tests and examples." OFF)
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
if(ONNX_LIGHT_ORT_BUILD_PYTHON OR ONNX_LIGHT_ORT_TEST_PYTHON OR ONNX_LIGHT_ORT_TEST_WITH_ORT)
  find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
  set(Python_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH
    "Python interpreter for onnx-light bindings." FORCE)
endif()

include(cmake/Dependencies.cmake)
set(onnx_light_ort_python_environment
  "PYTHONPATH=path_list_prepend:${ONNX_LIGHT_ORT_PYTHON_DIR}"
  "PYTHONPATH=path_list_prepend:${PROJECT_SOURCE_DIR}")
if(WIN32)
  list(APPEND onnx_light_ort_python_environment
    "PATH=path_list_prepend:${ONNX_LIGHT_ORT_PYTHON_DIR}/onnx_light/onnx_py")
endif()
set(onnx_light_ort_python_test_command "${Python3_EXECUTABLE}")
if(WIN32)
  list(APPEND onnx_light_ort_python_test_command
    "${PROJECT_SOURCE_DIR}/cmake/run_python_with_dll_paths.py")
endif()

if(ONNX_LIGHT_ORT_BUILD_TESTS)
  enable_testing()
  include(cmake/Testing.cmake)
endif()

if(ONNX_LIGHT_ORT_BUILD_HOST)
  add_subdirectory(onnx_light_ort/common_ort_to_light)
  add_subdirectory(onnx_light_ort/bridge_ops_ort_called_by_light)
  add_subdirectory(onnx_light_ort/bridge_ep_ort_called_by_light)
endif()

if(ONNX_LIGHT_ORT_BUILD_PLUGIN)
  add_subdirectory(onnx_light_ort/common_light_to_ort)
  add_subdirectory(onnx_light_ort/bridge_ops_light_called_by_ort)
  add_subdirectory(onnx_light_ort/bridge_ep_light_called_by_ort)
  add_subdirectory(onnx_light_ort/custom_ops)
endif()

if(ONNX_LIGHT_ORT_BUILD_TESTS)
  if(ONNX_LIGHT_ORT_TEST_WITH_ORT)
    add_test(NAME custom_ops_light_called_by_ort_python
      COMMAND ${onnx_light_ort_python_test_command} "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_custom_ops_light_called_by_ort.py"
        --library "$<TARGET_FILE:onnx_light_ort_custom_ops>"
    )
    set_tests_properties(custom_ops_light_called_by_ort_python PROPERTIES TIMEOUT 120)
    set_tests_properties(custom_ops_light_called_by_ort_python PROPERTIES
      LABELS "integration;custom_ops;bridge_ops_light_called_by_ort;common_light_to_ort"
      ENVIRONMENT_MODIFICATION "${onnx_light_ort_python_environment}")
  endif()
endif()

if(ONNX_LIGHT_ORT_BUILD_PLUGIN AND ONNX_LIGHT_ORT_BUILD_EXAMPLES)
  add_subdirectory(examples/custom_ops)
endif()

if(ONNX_LIGHT_ORT_TEST_PYTHON AND ONNX_LIGHT_ORT_TEST_WITH_ORT AND
   ONNX_LIGHT_ORT_BUILD_EXAMPLES)
  add_test(NAME gallery_custom_ops_light_called_by_ort
    COMMAND ${onnx_light_ort_python_test_command} "${PROJECT_SOURCE_DIR}/docs/examples/plot_custom_ops_light_called_by_ort.py")
  set_tests_properties(gallery_custom_ops_light_called_by_ort PROPERTIES LABELS "example;python;custom_ops"
    TIMEOUT 60 ENVIRONMENT "ONNX_LIGHT_ORT_PLUGIN_LIBRARY=$<TARGET_FILE:onnx_light_ort_custom_ops>"
    ENVIRONMENT_MODIFICATION "${onnx_light_ort_python_environment}")
  add_test(NAME gallery_custom_ops_ort_called_by_light
    COMMAND ${onnx_light_ort_python_test_command} "${PROJECT_SOURCE_DIR}/docs/examples/plot_custom_ops_ort_called_by_light.py")
  set_tests_properties(gallery_custom_ops_ort_called_by_light PROPERTIES LABELS "example;python;custom_ops;bridge_ops_ort_called_by_light"
    TIMEOUT 60 ENVIRONMENT
    "ONNX_LIGHT_ORT_HOST_LIBRARY=$<TARGET_FILE:onnx_light_ort_bridge_ops_ort_called_by_light>;ONNX_LIGHT_ORT_INCOMING_CUSTOM_OP_LIBRARY=$<TARGET_FILE:onnx_light_ort_ort_custom_ops>"
    ENVIRONMENT_MODIFICATION "${onnx_light_ort_python_environment}")
endif()
