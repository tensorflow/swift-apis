add_library(CTensorFlow INTERFACE)
if(NOT USE_BUNDLED_CTENSORFLOW)
  target_include_directories(CTensorFlow INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR})
endif()
target_link_directories(CTensorFlow INTERFACE
  $<TARGET_FILE_DIR:x10>)
target_link_libraries(CTensorFlow INTERFACE
  x10)

list(GET X10_INCLUDE_DIRS 0 X10_SOURCE_DIR)

install(FILES module.modulemap CTensorFlow.h
  DESTINATION lib/swift/tensorflow)
install(DIRECTORY ${X10_SOURCE_DIR}/tensorflow/c
  DESTINATION lib/swift/tensorflow
  FILES_MATCHING
    PATTERN c_api.h
    PATTERN c_api_experimental.h
    PATTERN tf_attrtype.h
    PATTERN tf_datatype.h
    PATTERN tf_status.h
    PATTERN tf_tensor.h
    PATTERN eager/c_api.h)
