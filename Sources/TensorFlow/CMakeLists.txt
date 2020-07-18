add_library(TensorFlow SHARED
  Exports.swift

  Bindings/RawOpsGenerated.swift
  Bindings/TFTensorOperation.swift
  Bindings/RawOpsAugmented.swift
  Bindings/RawOpsDispatching.swift

  Core/DataTypes.swift
  Core/Execution.swift
  Core/Tensor.swift
  Core/TensorHandle.swift
  Core/Utilities.swift

  Operators/Math.swift
)
target_sources(TensorFlow PRIVATE
  ../x10/swift_bindings/Device.swift
  )
target_compile_definitions(TensorFlow PRIVATE
  USING_X10_BACKEND
  DEFAULT_BACKEND_EAGER)
target_link_libraries(TensorFlow PRIVATE
  CX10
  x10)
set_target_properties(TensorFlow PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY}
  OUTPUT_NAME swiftTensorFlow)
target_compile_options(TensorFlow PRIVATE
  $<$<BOOL:${BUILD_TESTING}>:-enable-testing>)
target_link_libraries(TensorFlow PUBLIC
  CTensorFlow
  $<$<NOT:$<PLATFORM_ID:Darwin>>:dispatch>
  $<$<NOT:$<PLATFORM_ID:Darwin>>:Foundation>
  Tensor)

_install_target(TensorFlow)
