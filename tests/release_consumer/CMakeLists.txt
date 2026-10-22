cmake_minimum_required(VERSION 3.28)
project(onnx_light_ort_release_consumer LANGUAGES CXX)
find_package(onnx_light_ort 0.2 CONFIG REQUIRED)
add_executable(release_consumer
  "../../onnx_light_ort/bridge_ops_ort_called_by_light/tests/test_custom_ops_ort_called_by_light_consumer.cc")
target_link_libraries(release_consumer PRIVATE onnx_light_ort::bridge_ops_ort_called_by_light)
enable_testing()
add_test(NAME release_consumer COMMAND release_consumer)
if(WIN32)
  set_tests_properties(release_consumer PROPERTIES ENVIRONMENT_MODIFICATION
    "PATH=path_list_prepend:$<TARGET_FILE_DIR:onnx_light_ort::bridge_ops_ort_called_by_light>")
endif()
