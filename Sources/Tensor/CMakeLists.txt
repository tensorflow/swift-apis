add_library(Tensor SHARED
  Random.swift
  TensorUtilities.swift)
set_target_properties(Tensor PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_compile_options(Tensor PRIVATE
  $<$<BOOL:${BUILD_TESTING}>:-enable-testing>)

_install_target(Tensor)
