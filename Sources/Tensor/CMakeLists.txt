add_library(Tensor SHARED
  Random.swift
  TensorUtilities.swift)
set_target_properties(Tensor PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})

get_swift_host_arch(swift_arch)
get_swift_host_os(swift_os)
install(TARGETS Tensor
  ARCHIVE DESTINATION lib/swift/${swift_os}
  LIBRARY DESTINATION lib/swift/${swift_os}
  RUNTIME DESTINATION bin)
if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
  install(FILES $<TARGET_PROPERTY:Tensor,Swift_MODULE_DIRECTORY>/Tensor.swiftdoc
    DESTINATION lib/swift/${swift_os}/Tensor.swiftmodule
    RENAME ${swift_arch}.swiftdoc)
  install(FILES $<TARGET_PROPERTY:Tensor,Swift_MODULE_DIRECTORY>/Tensor.swiftmodule
    DESTINATION lib/swift/${swift_os}/Tensor.swiftmodule
    RENAME ${swift_arch}.swiftmodule)
else()
  install(FILES
    $<TARGET_PROPERTY:Tensor,Swift_MODULE_DIRECTORY>/Tensor.swiftdoc
    $<TARGET_PROPERTY:Tensor,Swift_MODULE_DIRECTORY>/Tensor.swiftmodule
    DESTINATION lib/swift/${swift_os}/${swift_arch})
endif()
