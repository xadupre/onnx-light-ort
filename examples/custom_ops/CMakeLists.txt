add_executable(onnx_light_ort_custom_ops_example custom_ops.cc)
target_link_libraries(onnx_light_ort_custom_ops_example PRIVATE onnx_light_ort_api ${CMAKE_DL_LIBS})
add_dependencies(onnx_light_ort_custom_ops_example onnx_light_ort_custom_ops)
if(MSVC)
  target_compile_options(onnx_light_ort_custom_ops_example PRIVATE /W4 /WX)
else()
  target_compile_options(onnx_light_ort_custom_ops_example PRIVATE -Wall -Wextra -Werror)
endif()

if(ONNX_LIGHT_ORT_TEST_WITH_ORT)
  set(model "${CMAKE_CURRENT_BINARY_DIR}/custom_ops_example.onnx")
  add_test(NAME custom_ops_python_example
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/custom_ops.py"
      --library "$<TARGET_FILE:onnx_light_ort_custom_ops>" --model-output "${model}")
  set_tests_properties(custom_ops_python_example PROPERTIES
    LABELS "example;custom_ops" TIMEOUT 60 FIXTURES_SETUP custom_ops_example_model
    ENVIRONMENT_MODIFICATION "${onnx_light_ort_python_environment}")
  if(ONNX_LIGHT_ORT_RUNTIME_LIBRARY)
    if(NOT EXISTS "${ONNX_LIGHT_ORT_RUNTIME_LIBRARY}")
      message(FATAL_ERROR "ORT runtime library does not exist: ${ONNX_LIGHT_ORT_RUNTIME_LIBRARY}")
    endif()
    add_test(NAME custom_ops_cpp_example
      COMMAND onnx_light_ort_custom_ops_example "${ONNX_LIGHT_ORT_RUNTIME_LIBRARY}"
        "$<TARGET_FILE:onnx_light_ort_custom_ops>" "${model}")
    set_tests_properties(custom_ops_cpp_example PROPERTIES
      LABELS "example;custom_ops" TIMEOUT 60 FIXTURES_REQUIRED custom_ops_example_model)
  else()
    message(STATUS "Set ONNX_LIGHT_ORT_RUNTIME_LIBRARY to also run the C++ example in CTest.")
  endif()
endif()
