# NOTE: this uses an underscored name here and changes that because
# `Experimental` is a utility library used to indicate that this is an
# experimental build.  This renaming is a quick workaround for that conflict.
add_library(_Experimental
  Complex.swift)
set_target_properties(_Experimental PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY}
  OUTPUT_NAME Experimental
  Swift_MODULE_NAME Experimental)
target_compile_options(_Experimental PRIVATE
  $<$<BOOL:${BUILD_TESTING}>:-enable-testing>)

_install_target(_Experimental)
