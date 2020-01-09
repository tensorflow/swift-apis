add_library(CTensorFlow INTERFACE)
target_include_directories(CTensorFlow INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(CTensorFlow INTERFACE
  tensorflow)
