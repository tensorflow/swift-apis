add_library(Tensor SHARED
  Random.swift
  TensorUtilities.swift)
set_target_properties(Tensor PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})

_install_target(Tensor)
