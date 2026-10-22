include(FetchContent)

set(ONNX_LIGHT_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(ONNX_LIGHT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ONNX_LIGHT_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(ONNX_LIGHT_BUILD_FUZZERS "${ONNX_LIGHT_ORT_BUILD_FUZZERS}" CACHE BOOL "" FORCE)
set(ONNX_LIGHT_BUILD_KERNELS OFF CACHE BOOL "" FORCE)
set(ONNX_LIGHT_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  onnx_light
  GIT_REPOSITORY https://github.com/xadupre/onnx-light.git
  GIT_TAG main
  GIT_SHALLOW TRUE
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(onnx_light)

find_package(Git REQUIRED)
execute_process(
  COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
  WORKING_DIRECTORY "${onnx_light_SOURCE_DIR}"
  OUTPUT_VARIABLE onnx_light_revision
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
message(STATUS "onnx-light main revision: ${onnx_light_revision}")

set(_ort_revision db5f5eaff7c32e09e90520374ac19b78d89094ae)
set(_ort_include "${CMAKE_CURRENT_BINARY_DIR}/ort-headers")
file(MAKE_DIRECTORY "${_ort_include}")

function(onnx_light_ort_fetch_header name expected_hash)
  set(destination "${_ort_include}/${name}")
  if(EXISTS "${destination}")
    file(SHA256 "${destination}" actual_hash)
    if(actual_hash STREQUAL expected_hash)
      return()
    endif()
  endif()
  file(DOWNLOAD
    "https://raw.githubusercontent.com/microsoft/onnxruntime/${_ort_revision}/include/onnxruntime/core/session/${name}"
    "${destination}"
    EXPECTED_HASH "SHA256=${expected_hash}"
    TLS_VERIFY ON
    STATUS download_status
    TIMEOUT 120
  )
  list(GET download_status 0 status_code)
  if(NOT status_code EQUAL 0)
    message(FATAL_ERROR "Cannot download ${name}: ${download_status}")
  endif()
endfunction()

onnx_light_ort_fetch_header(
  onnxruntime_c_api.h 28b7bf042aec952a7e20769409f26d642b6e7a9eea04ad8cba4883123a4c727d
)
onnx_light_ort_fetch_header(
  onnxruntime_ep_c_api.h 988ea0f2aa26ea96f107ba06826480792365b392a3659aadff804845a2080a29
)
onnx_light_ort_fetch_header(
  onnxruntime_error_code.h 5ce3b054e798eced8d14f5b86e98692fd33470463f96194ce0700a2d53dd8721
)

add_library(onnx_light_ort_api INTERFACE)
target_include_directories(onnx_light_ort_api SYSTEM INTERFACE "${_ort_include}")
target_compile_features(onnx_light_ort_api INTERFACE cxx_std_20)
