add_library(Tensor SHARED
  Random.swift
  TensorUtilities.swift)
set_target_properties(Tensor PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})

install(TARGETS Tensor
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)
get_swift_host_arch(swift_arch)
install(FILES
  $<TARGET_PROPERTY:Tensor,Swift_MODULE_DIRECTORY>/Tensor.swiftdoc
  $<TARGET_PROPERTY:Tensor,Swift_MODULE_DIRECTORY>/Tensor.swiftmodule
  DESTINATION lib/swift/${swift_arch})
