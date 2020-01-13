find_path(TensorFlow_INCLUDE_DIR
          NAMES
            tensorflow.h
          HINTS
            ${CMAKE_INSTALL_FULL_INCLUDEDIR})
find_library(TensorFlow_LIBRARY
             NAMES
               tensorflow
             HINTS
               ${CMAKE_INSTALL_FULL_LIBDIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorFlow
  FOUND_VAR
    TensorFlow_FOUND
  REQUIRED_VARS
    TensorFlow_INCLUDE_DIR
    TensorFlow_LIBRARY
  VERSION_VAR
    TensorFlow_VERSION)

if(TensorFlow_FOUND AND NOT TARGET tensorflow)
  add_library(tensorflow UNKNOWN IMPORTED)
  set_target_properties(tensorflow PROPERTIES
    IMPORTED_LOCATION ${TensorFlow_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${TensorFlow_INCLUDE_DIR})
endif()

mark_as_advanced(TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)
