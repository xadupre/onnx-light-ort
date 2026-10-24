# Build the upstream Python extensions against the same runtime as the bridges.
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE _nanobind_dir OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
find_package(nanobind CONFIG REQUIRED PATHS "${_nanobind_dir}")

file(GLOB_RECURSE _cpu_registration_sources CONFIGURE_DEPENDS
  "${onnx_light_cpu_SOURCE_DIR}/onnx_light_cpu/kernels/*.cc"
  "${onnx_light_cpu_SOURCE_DIR}/onnx_light_cpu/gradient/*.cc"
  "${onnx_light_cpu_SOURCE_DIR}/onnx_light_cpu/patterns/*.cc"
  "${onnx_light_cpu_SOURCE_DIR}/onnx_light_cpu/schemas/*.cc"
  "${onnx_light_cpu_SOURCE_DIR}/onnx_light_cpu/shapes/*.cc")
add_library(onnx_light_ort_cpu_kernels STATIC ${_cpu_registration_sources})
set_target_properties(onnx_light_ort_cpu_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(onnx_light_ort_cpu_kernels PUBLIC "${onnx_light_cpu_SOURCE_DIR}")
target_link_libraries(onnx_light_ort_cpu_kernels PUBLIC
  ml_cpu_compute::ml_cpu_compute lib_onnx_core lib_onnx_kernels lib_onnx_shape)
target_compile_definitions(onnx_light_ort_cpu_kernels PUBLIC ONNX_LIGHT_CPU_HAS_ONNX_SHAPE=1)
foreach(feature AVX512BF16 AVX2_FMA F16C)
  if(ML_CPU_COMPUTE_HAVE_${feature}_FLAG)
    target_compile_definitions(onnx_light_ort_cpu_kernels PRIVATE
      ML_CPU_COMPUTE_HAVE_${feature}=1)
  endif()
endforeach()

file(GLOB _cpu_python_sources CONFIGURE_DEPENDS
  "${onnx_light_cpu_SOURCE_DIR}/onnx_light_cpu/onnx_py/*.cc")
file(GLOB _cpu_register_sources CONFIGURE_DEPENDS
  "${onnx_light_cpu_SOURCE_DIR}/onnx_light_cpu/onnx_py/_cpuregister*.cc")
list(REMOVE_ITEM _cpu_python_sources ${_cpu_register_sources})
nanobind_add_module(_cpukernels STABLE_ABI ${_cpu_python_sources})
target_include_directories(_cpukernels PRIVATE "${onnx_light_cpu_SOURCE_DIR}")
target_link_libraries(_cpukernels PRIVATE ml_cpu_compute::ml_cpu_compute)
nanobind_add_module(_cpuregister STABLE_ABI ${_cpu_register_sources})
target_link_libraries(_cpuregister PRIVATE onnx_light_ort_cpu_kernels)

set(_cpu_python_dir "${PROJECT_BINARY_DIR}/python")
file(COPY "${onnx_light_cpu_SOURCE_DIR}/onnx_light_cpu"
  DESTINATION "${_cpu_python_dir}" FILES_MATCHING PATTERN "*.py")
foreach(target _cpukernels _cpuregister)
  set_target_properties("${target}" PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${_cpu_python_dir}/onnx_light_cpu/onnx_py"
    RUNTIME_OUTPUT_DIRECTORY "${_cpu_python_dir}/onnx_light_cpu/onnx_py")
  foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER "${config}" config_upper)
    set_target_properties("${target}" PROPERTIES
      "LIBRARY_OUTPUT_DIRECTORY_${config_upper}" "${_cpu_python_dir}/onnx_light_cpu/onnx_py"
      "RUNTIME_OUTPUT_DIRECTORY_${config_upper}" "${_cpu_python_dir}/onnx_light_cpu/onnx_py")
  endforeach()
endforeach()
add_dependencies(onnx_light_ort_python _cpukernels _cpuregister)
