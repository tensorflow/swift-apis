
if(DEFINED CMAKE_Swift_COMPILER)
  get_filename_component(_Swift_TOOLCHAIN ${CMAKE_Swift_COMPILER} DIRECTORY)
  get_filename_component(_Swift_TOOLCHAIN ${_Swift_TOOLCHAIN} DIRECTORY)
  get_filename_component(_Swift_TOOLCHAIN ${_Swift_TOOLCHAIN} DIRECTORY)
  string(TOLOWER ${CMAKE_SYSTEM_NAME} system_lc)
  file(TO_CMAKE_PATH ${_Swift_TOOLCHAIN}/usr/lib/swift/${system_lc} _Swift_LIBDIR)
endif()

find_library(X10_LIBRARY
  NAMES x10
  HINTS ${CMAKE_INSTALL_FULL_LIBDIR} ${_Swift_LIBDIR})

unset(_Swift_TOOLCHAIN)
unset(_Swift_LIBDIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(X10
  FOUND_VAR X10_FOUND
  REQUIRED_VARS X10_LIBRARY)

if(X10_FOUND AND NOT TARGET x10)
  add_library(x10 UNKNOWN IMPORTED)
  set_target_properties(x10 PROPERTIES
    IMPORTED_LOCATION ${X10_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES "${X10_INCLUDE_DIRS}")
endif()

mark_as_advanced(X10_INCLUDE_DIRS X10_LIBRARY)
