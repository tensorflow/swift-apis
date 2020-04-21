add_library(CX10 INTERFACE)
target_include_directories(CX10 INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_directories(CX10 INTERFACE
  $<TARGET_FILE_DIR:x10>)
target_link_libraries(CX10 INTERFACE
  x10)
