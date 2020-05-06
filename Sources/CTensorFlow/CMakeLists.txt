add_library(CTensorFlow INTERFACE)
if(NOT USE_BUNDLED_CTENSORFLOW)
  target_include_directories(CTensorFlow INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR})
endif()
if(BUILD_X10)
  target_link_directories(CTensorFlow INTERFACE
    $<TARGET_FILE_DIR:x10>)
  target_link_libraries(CTensorFlow INTERFACE
    x10)
else()
  target_link_directories(CTensorFlow INTERFACE
    $<TARGET_FILE_DIR:tensorflow>)
  target_link_libraries(CTensorFlow INTERFACE
    tensorflow)
endif()

install(FILES module.modulemap CTensorFlow.h
  DESTINATION lib/swift/tensorflow)
install(DIRECTORY $<TARGET_PROPERTY:tensorflow,INTERFACE_INCLUDE_DIRECTORIES>/tensorflow/c
  DESTINATION lib/swift/tensorflow
  FILES_MATCHING
    PATTERN c_api.h
    PATTERN c_api_experimental.h
    PATTERN tf_attrtype.h
    PATTERN tf_datatype.h
    PATTERN tf_status.h
    PATTERN tf_tensor.h
    PATTERN eager/c_api.h)
