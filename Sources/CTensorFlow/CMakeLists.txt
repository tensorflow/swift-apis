add_library(CTensorFlow INTERFACE)
if(NOT USE_BUNDLED_CTENSORFLOW)
  target_include_directories(CTensorFlow INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()
target_link_directories(CTensorFlow INTERFACE
  $<TARGET_FILE_DIR:x10>)
target_link_libraries(CTensorFlow INTERFACE
  x10)

list(GET X10_INCLUDE_DIRS 0 X10_SOURCE_DIR)

install(FILES include/module.modulemap include/CTensorFlow.h
  DESTINATION lib/swift/tensorflow)
install(FILES
  ${X10_SOURCE_DIR}/tensorflow/c/c_api.h
  ${X10_SOURCE_DIR}/tensorflow/c/c_api_experimental.h
  ${X10_SOURCE_DIR}/tensorflow/c/c_api_macros.h
  ${X10_SOURCE_DIR}/tensorflow/c/tf_attrtype.h
  ${X10_SOURCE_DIR}/tensorflow/c/tf_datatype.h
  ${X10_SOURCE_DIR}/tensorflow/c/tf_file_statistics.h
  ${X10_SOURCE_DIR}/tensorflow/c/tf_status.h
  ${X10_SOURCE_DIR}/tensorflow/c/tf_tensor.h
  ${X10_SOURCE_DIR}/tensorflow/c/tf_tstring.h
  DESTINATION lib/swift/tensorflow/c)
install(FILES
  ${X10_SOURCE_DIR}/tensorflow/c/eager/c_api.h
  DESTINATION lib/swift/tensorflow/c/eager)
install(FILES
  ${X10_SOURCE_DIR}/tensorflow/core/platform/ctstring.h
  ${X10_SOURCE_DIR}/tensorflow/core/platform/ctstring_internal.h
  DESTINATION lib/swift/tensorflow/core/platform)
