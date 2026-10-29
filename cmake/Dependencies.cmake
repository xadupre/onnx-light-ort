include(FetchContent)

set(_onnx_light_python OFF)
if(ONNX_LIGHT_ORT_BUILD_PYTHON OR ONNX_LIGHT_ORT_TEST_PYTHON OR ONNX_LIGHT_ORT_TEST_WITH_ORT)
  set(_onnx_light_python ON)
endif()
set(ONNX_LIGHT_BUILD_PYTHON "${_onnx_light_python}" CACHE BOOL "" FORCE)
set(ONNX_LIGHT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ONNX_LIGHT_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(ONNX_LIGHT_BUILD_FUZZERS "${ONNX_LIGHT_ORT_BUILD_FUZZERS}" CACHE BOOL "" FORCE)
set(_native_cpu_python OFF)
if(ONNX_LIGHT_ORT_BUILD_PLUGIN AND
    (ONNX_LIGHT_ORT_TEST_WITH_ORT OR
     (ONNX_LIGHT_ORT_BUILD_PYTHON AND ONNX_LIGHT_ORT_BUILD_EXAMPLES)))
  set(_native_cpu_python ON)
endif()
set(ONNX_LIGHT_BUILD_KERNELS "${_native_cpu_python}" CACHE BOOL "" FORCE)
set(ONNX_LIGHT_INSTALL OFF CACHE BOOL "" FORCE)

set(ONNX_LIGHT_ORT_ONNX_LIGHT_SOURCE_DIR "" CACHE PATH "Local onnx-light runtime source checkout.")
if(ONNX_LIGHT_ORT_ONNX_LIGHT_SOURCE_DIR)
  include("${CMAKE_CURRENT_LIST_DIR}/LocalOnnxLight.cmake")
else()
  set(ONNX_LIGHT_ORT_ONNX_LIGHT_REVISION "0.2.2" CACHE STRING "onnx-light Git revision to build.")
  set(_onnx_light_shallow OFF)
  if(ONNX_LIGHT_ORT_ONNX_LIGHT_REVISION STREQUAL "main")
    set(_onnx_light_shallow ON)
  endif()
  FetchContent_Declare(
    onnx_light
    GIT_REPOSITORY https://github.com/xadupre/onnx-light.git
    GIT_TAG "${ONNX_LIGHT_ORT_ONNX_LIGHT_REVISION}"
    GIT_SHALLOW "${_onnx_light_shallow}"
    SYSTEM
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(onnx_light)

  set(ONNX_LIGHT_ORT_PYTHON_DIR "${PROJECT_BINARY_DIR}/python")
  if(_onnx_light_python)
    file(COPY "${onnx_light_SOURCE_DIR}/onnx_light"
      DESTINATION "${ONNX_LIGHT_ORT_PYTHON_DIR}"
      FILES_MATCHING PATTERN "*.py")
    set(_python_library_dir "${ONNX_LIGHT_ORT_PYTHON_DIR}/onnx_light/onnx_py")
    set(_model_python_targets _onnxpyprotoop _onnxpyprotolib
        lib_onnx_proto lib_onnx_threads lib_onnx_core lib_onnx_manipulations
        lib_onnx_op lib_onnx_lib lib_onnx_shape lib_onnx_patterns)
    if(_native_cpu_python)
      list(APPEND _model_python_targets
        _onnxpycore _onnxpykernels lib_onnx_kernels)
    endif()
    foreach(target IN LISTS _model_python_targets)
      set_target_properties("${target}" PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${_python_library_dir}"
        RUNTIME_OUTPUT_DIRECTORY "${_python_library_dir}")
      foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
        string(TOUPPER "${config}" config_upper)
        set_target_properties("${target}" PROPERTIES
          "LIBRARY_OUTPUT_DIRECTORY_${config_upper}" "${_python_library_dir}"
          "RUNTIME_OUTPUT_DIRECTORY_${config_upper}" "${_python_library_dir}")
      endforeach()
    endforeach()
    add_custom_target(onnx_light_ort_python ALL DEPENDS _onnxpyprotoop _onnxpyprotolib)
    if(_native_cpu_python)
      add_dependencies(onnx_light_ort_python _onnxpycore _onnxpykernels)
    endif()
  endif()
endif()

find_package(Git REQUIRED)
execute_process(
  COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
  WORKING_DIRECTORY "${onnx_light_SOURCE_DIR}"
  OUTPUT_VARIABLE onnx_light_revision
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
message(STATUS "onnx-light revision: ${onnx_light_revision}")

if(ONNX_LIGHT_ORT_BUILD_PLUGIN)
  FetchContent_Declare(
    onnx_light_cpu
    GIT_REPOSITORY https://github.com/xadupre/onnx-light-cpu.git
    GIT_TAG 0.2.1.1
    SOURCE_SUBDIR ml_cpu_compute
    SYSTEM
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(onnx_light_cpu)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${onnx_light_cpu_SOURCE_DIR}"
    OUTPUT_VARIABLE onnx_light_cpu_revision
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  message(STATUS "onnx-light-cpu revision: ${onnx_light_cpu_revision}")
  if(_native_cpu_python)
    include("${CMAKE_CURRENT_LIST_DIR}/NativeCpuPython.cmake")
  endif()
endif()

set(_ort_revision db5f5eaff7c32e09e90520374ac19b78d89094ae)
set(_ort_include "${CMAKE_CURRENT_BINARY_DIR}/ort-headers")
file(MAKE_DIRECTORY "${_ort_include}")

function(onnx_light_ort_fetch_file name expected_hash)
  set(destination "${_ort_include}/${name}")
  if(EXISTS "${destination}")
    file(SHA256 "${destination}" actual_hash)
    if(actual_hash STREQUAL expected_hash)
      return()
    endif()
  endif()
  set(source_path "include/onnxruntime/core/session/${name}")
  if(ARGC EQUAL 3)
    set(source_path "${ARGV2}")
  endif()
  file(DOWNLOAD
    "https://raw.githubusercontent.com/microsoft/onnxruntime/${_ort_revision}/${source_path}"
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

onnx_light_ort_fetch_file(
  onnxruntime_c_api.h 28b7bf042aec952a7e20769409f26d642b6e7a9eea04ad8cba4883123a4c727d
)
onnx_light_ort_fetch_file(
  onnxruntime_ep_c_api.h 988ea0f2aa26ea96f107ba06826480792365b392a3659aadff804845a2080a29
)
onnx_light_ort_fetch_file(
  onnxruntime_error_code.h 5ce3b054e798eced8d14f5b86e98692fd33470463f96194ce0700a2d53dd8721
)
onnx_light_ort_fetch_file(
  LICENSE 2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c LICENSE
)

add_library(onnx_light_ort_api INTERFACE)
target_include_directories(onnx_light_ort_api SYSTEM INTERFACE
  "$<BUILD_INTERFACE:${_ort_include}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_compile_features(onnx_light_ort_api INTERFACE cxx_std_20)
