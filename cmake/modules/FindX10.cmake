find_path(X10_INCLUDE_DIR
          NAMES
            device_wrapper.h
            xla_tensor_tf_ops.h
            xla_tensor_wrapper.h
          HINTS
            ${CMAKE_INSTALL_FULL_INCLUDEDIR})
find_library(X10_LIBRARY
             NAMES
               x10
             HINTS
               ${CMAKE_INSTALL_FULL_LIBDIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(X10
  FOUND_VAR
    X10_FOUND
  REQUIRED_VARS
    X10_INCLUDE_DIR
    X10_LIBRARY
  VERSION_VAR
    X10_VERSION)

if(X10_FOUND AND NOT TARGET x10)
  add_library(x10 UNKNOWN IMPORTED)
  set_target_properties(x10 PROPERTIES
    IMPORTED_LOCATION ${X10_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${X10_INCLUDE_DIR})
endif()

mark_as_advanced(X10_INCLUDE_DIR X10_LIBRARY)
