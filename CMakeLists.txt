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
  add_library(onnx_light_ort_host INTERFACE)
  add_library(onnx_light_ort::host ALIAS onnx_light_ort_host)
  target_link_libraries(onnx_light_ort_host INTERFACE lib_onnx_core onnx_light_ort_api)
endif()

if(ONNX_LIGHT_ORT_BUILD_PLUGIN)
  add_library(onnx_light_ort_plugin SHARED
    src/custom_ops.cc
    "${onnx_light_SOURCE_DIR}/onnx_light/onnx_extensions/kernels/kernels/math/kernel_leakyrelu.cc"
  )
  add_library(onnx_light_ort::plugin ALIAS onnx_light_ort_plugin)
  target_link_libraries(onnx_light_ort_plugin PRIVATE lib_onnx_core onnx_light_ort_api)
  set_target_properties(onnx_light_ort_plugin PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )
  if(APPLE)
    target_link_options(onnx_light_ort_plugin PRIVATE "LINKER:-exported_symbol,_RegisterCustomOps")
  elseif(UNIX)
    target_link_options(onnx_light_ort_plugin PRIVATE "LINKER:--exclude-libs,ALL")
  endif()
  if(MSVC)
    target_compile_options(onnx_light_ort_plugin PRIVATE /W4 /WX)
  else()
    target_compile_options(onnx_light_ort_plugin PRIVATE -Wall -Wextra -Werror)
  endif()
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
    target_link_libraries(onnx_light_ort_dependency_test PRIVATE onnx_light_ort::host)
  endif()
  if(ONNX_LIGHT_ORT_BUILD_PLUGIN)
    add_dependencies(onnx_light_ort_dependency_test onnx_light_ort_plugin)
  endif()
  add_test(NAME dependency_contract COMMAND onnx_light_ort_dependency_test)
  set_tests_properties(dependency_contract PROPERTIES TIMEOUT 60)
  if(ONNX_LIGHT_ORT_TEST_WITH_ORT)
    find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
    add_test(NAME ort_custom_ops
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ort_custom_ops.py"
        --library "$<TARGET_FILE:onnx_light_ort_plugin>"
    )
    set_tests_properties(ort_custom_ops PROPERTIES TIMEOUT 120)
  endif()
endif()
