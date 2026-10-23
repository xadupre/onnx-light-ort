if(ONNX_LIGHT_ORT_INSTALL)
  message(FATAL_ERROR "Local onnx-light runtime builds require ONNX_LIGHT_ORT_INSTALL=OFF.")
endif()
if(ONNX_LIGHT_ORT_BUILD_FUZZERS)
  message(FATAL_ERROR "Local onnx-light runtime builds do not build upstream fuzzers.")
endif()

set(onnx_light_SOURCE_DIR "${ONNX_LIGHT_ORT_ONNX_LIGHT_SOURCE_DIR}")
set(ONNX_LIGHT_ORT_PYTHON_DIR "${ONNX_LIGHT_ORT_ONNX_LIGHT_PYTHON_DIR}")
if(_onnx_light_python AND NOT EXISTS "${ONNX_LIGHT_ORT_PYTHON_DIR}/onnx_light/__init__.py")
  message(FATAL_ERROR "ONNX_LIGHT_ORT_ONNX_LIGHT_PYTHON_DIR must contain the local model package.")
endif()
if(_onnx_light_python)
  add_custom_target(onnx_light_ort_python ALL)
endif()

find_package(Threads REQUIRED)
foreach(component core proto threads)
  string(TOUPPER "${component}" component_upper)
  set(prefix "ONNX_LIGHT_ORT_ONNX_LIGHT_${component_upper}")
  if(NOT EXISTS "${${prefix}_LIBRARY}" OR IS_DIRECTORY "${${prefix}_LIBRARY}")
    message(FATAL_ERROR "${prefix}_LIBRARY must name the shared library loaded by Python.")
  endif()
  add_library("lib_onnx_${component}" SHARED IMPORTED GLOBAL)
  set_target_properties("lib_onnx_${component}" PROPERTIES
    IMPORTED_LOCATION "${${prefix}_LIBRARY}"
    INTERFACE_COMPILE_FEATURES cxx_std_20)
  if(WIN32)
    if(NOT EXISTS "${${prefix}_IMPLIB}" OR IS_DIRECTORY "${${prefix}_IMPLIB}")
      message(FATAL_ERROR "${prefix}_IMPLIB must name the matching Windows import library.")
    endif()
    set_target_properties("lib_onnx_${component}" PROPERTIES
      IMPORTED_IMPLIB "${${prefix}_IMPLIB}")
  endif()
endforeach()

if(_native_cpu_python)
  foreach(component kernels shape)
    find_library(_runtime_${component}
      NAMES "lib_onnx_${component}"
      PATHS "${ONNX_LIGHT_ORT_PYTHON_DIR}/onnx_light/onnx_py"
      NO_DEFAULT_PATH NO_CACHE REQUIRED)
    add_library("lib_onnx_${component}" UNKNOWN IMPORTED GLOBAL)
    set_target_properties("lib_onnx_${component}" PROPERTIES
      IMPORTED_LOCATION "${_runtime_${component}}"
      INTERFACE_LINK_LIBRARIES lib_onnx_core)
  endforeach()
endif()

set_target_properties(lib_onnx_threads PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${onnx_light_SOURCE_DIR}/onnx_threads/include"
  INTERFACE_COMPILE_DEFINITIONS ONNX_THREADS_SHARED
  INTERFACE_LINK_LIBRARIES Threads::Threads)
set_target_properties(lib_onnx_proto PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES
    "${onnx_light_SOURCE_DIR}/onnx_proto/include;${onnx_light_SOURCE_DIR}/onnx_proto/include/onnx_proto;${onnx_light_SOURCE_DIR}/onnx_proto/helpers/include"
  INTERFACE_LINK_LIBRARIES lib_onnx_threads)
set_target_properties(lib_onnx_core PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${onnx_light_SOURCE_DIR}/onnx_light"
  INTERFACE_COMPILE_DEFINITIONS ONNX_LIGHT_SHARED_LIBS
  INTERFACE_LINK_LIBRARIES lib_onnx_proto)
message(STATUS "Using local onnx-light runtime: ${ONNX_LIGHT_ORT_ONNX_LIGHT_CORE_LIBRARY}")
