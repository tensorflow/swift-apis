add_library(Tensor SHARED
  Empty.swift
  )
set_target_properties(Tensor PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_compile_options(Tensor PRIVATE
  $<$<BOOL:${BUILD_TESTING}>:-enable-testing>)

_install_target(Tensor)
