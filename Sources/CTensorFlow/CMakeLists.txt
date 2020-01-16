add_library(CTensorFlow INTERFACE)
if(NOT USE_BUNDLED_CTENSORFLOW)
  target_include_directories(CTensorFlow INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR})
endif()
target_link_libraries(CTensorFlow INTERFACE
  tensorflow)
