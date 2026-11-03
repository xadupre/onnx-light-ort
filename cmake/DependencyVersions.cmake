file(READ "${CMAKE_CURRENT_LIST_DIR}/../pyproject.toml" _pyproject)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/../pyproject.toml")
string(REGEX MATCH "\\[tool\\.onnx-light-ort\\.dependencies\\][^[]*" _dependencies
  "${_pyproject}")
foreach(dependency IN ITEMS onnx-light onnx-light-cpu)
  string(REGEX MATCHALL "\n${dependency} = \"[0-9]+\\.[0-9]+\\.[0-9]+\""
    _matches "${_dependencies}")
  list(LENGTH _matches _count)
  if(NOT _count EQUAL 1)
    message(FATAL_ERROR "Expected one ${dependency} version in pyproject.toml")
  endif()
  string(REGEX REPLACE ".*\"([0-9.]+)\"" "\\1" _version "${_matches}")
  string(TOUPPER "${dependency}" _name)
  string(REPLACE "-" "_" _name "${_name}")
  set(ONNX_LIGHT_ORT_${_name}_VERSION "${_version}")
endforeach()
