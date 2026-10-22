include(CMakePackageConfigHelpers)

if(SKBUILD)
  if(NOT PROJECT_VERSION VERSION_EQUAL SKBUILD_PROJECT_VERSION)
    message(FATAL_ERROR "CMake and Python package versions must match.")
  endif()
  set(_library_dir "onnx_light_ort/_native")
  set(_binary_dir "${_library_dir}")
  set(_data_dir "${_library_dir}")
else()
  set(_library_dir "${CMAKE_INSTALL_LIBDIR}")
  set(_binary_dir "${CMAKE_INSTALL_BINDIR}")
  set(_data_dir "${CMAKE_INSTALL_DATADIR}/onnx-light-ort")
endif()

if(APPLE)
  set(_install_rpath "@loader_path")
elseif(UNIX)
  set(_install_rpath "$ORIGIN")
endif()

foreach(component custom_ops bridge_ops_ort_called_by_light)
  set(target "onnx_light_ort_${component}")
  if(TARGET "${target}")
    set_target_properties("${target}" PROPERTIES
      EXPORT_NAME "${component}" INSTALL_RPATH "${_install_rpath}")
    install(TARGETS "${target}" EXPORT onnx_light_ortTargets
      LIBRARY DESTINATION "${_library_dir}" COMPONENT Runtime
      RUNTIME DESTINATION "${_binary_dir}" COMPONENT Runtime
      ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Development)
  endif()
endforeach()

foreach(target lib_onnx_core lib_onnx_proto lib_onnx_threads ml_cpu_compute)
  if(TARGET "${target}")
    get_target_property(_type "${target}" TYPE)
    if(_type STREQUAL "SHARED_LIBRARY")
      if(SKBUILD)
        # Avoid loading an unrelated onnx-light model package's native libraries.
        set_target_properties("${target}" PROPERTIES OUTPUT_NAME "onnx_light_ort_${target}")
      endif()
      set_target_properties("${target}" PROPERTIES INSTALL_RPATH "${_install_rpath}")
      install(TARGETS "${target}"
        LIBRARY DESTINATION "${_library_dir}" COMPONENT Runtime
        RUNTIME DESTINATION "${_binary_dir}" COMPONENT Runtime)
    endif()
  endif()
endforeach()

file(WRITE "${PROJECT_BINARY_DIR}/release-build-info.txt"
  "onnx-light-ort: ${PROJECT_VERSION}\nonnx-light: ${onnx_light_revision}\nonnx-light-cpu: ${onnx_light_cpu_revision}\n")
install(FILES "${PROJECT_BINARY_DIR}/release-build-info.txt"
  DESTINATION "${_data_dir}" COMPONENT Runtime)
install(FILES "${PROJECT_SOURCE_DIR}/LICENSE"
  DESTINATION "${_data_dir}/licenses/onnx-light-ort" COMPONENT Runtime)
install(FILES "${onnx_light_SOURCE_DIR}/LICENSE"
  DESTINATION "${_data_dir}/licenses/onnx-light" COMPONENT Runtime)
install(FILES "${onnx_light_SOURCE_DIR}/onnx_proto/include/onnx_proto/dlpack/LICENSE"
  DESTINATION "${_data_dir}/licenses/dlpack" COMPONENT Runtime)
install(FILES "${_ort_include}/LICENSE"
  DESTINATION "${_data_dir}/licenses/onnxruntime" COMPONENT Runtime)
if(ONNX_LIGHT_ORT_BUILD_PLUGIN)
  install(FILES "${onnx_light_cpu_SOURCE_DIR}/ml_cpu_compute/LICENSE"
    DESTINATION "${_data_dir}/licenses/ml_cpu_compute" COMPONENT Runtime)
endif()

if(NOT SKBUILD)
  set_target_properties(onnx_light_ort_api PROPERTIES EXPORT_NAME api)
  install(TARGETS onnx_light_ort_api EXPORT onnx_light_ortTargets)
  install(FILES "${_ort_include}/onnxruntime_c_api.h"
    "${_ort_include}/onnxruntime_ep_c_api.h" "${_ort_include}/onnxruntime_error_code.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT Development)
  if(ONNX_LIGHT_ORT_BUILD_HOST)
    install(FILES "${PROJECT_SOURCE_DIR}/onnx_light_ort/bridge_ops_ort_called_by_light/host.h"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light_ort/bridge_ops_ort_called_by_light"
      COMPONENT Development)
  endif()
  set(_config_dir "${CMAKE_INSTALL_LIBDIR}/cmake/onnx_light_ort")
  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/onnx_light_ortConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/onnx_light_ortConfig.cmake"
    INSTALL_DESTINATION "${_config_dir}")
  write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/onnx_light_ortConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}" COMPATIBILITY SameMajorVersion)
  install(EXPORT onnx_light_ortTargets NAMESPACE onnx_light_ort::
    DESTINATION "${_config_dir}" COMPONENT Development)
  install(FILES "${PROJECT_BINARY_DIR}/onnx_light_ortConfig.cmake"
    "${PROJECT_BINARY_DIR}/onnx_light_ortConfigVersion.cmake"
    DESTINATION "${_config_dir}" COMPONENT Development)
endif()
