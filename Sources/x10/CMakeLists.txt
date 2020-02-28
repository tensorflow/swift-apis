include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(x10 IMPORTED UNKNOWN)
set_target_properties(x10 PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${X10_INCLUDE_DIR}
  IMPORTED_LOCATION ${TensorFlow_LIBRARY})

add_library(x10_c_wrappers STATIC
  swift_bindings/device_wrapper.cc
  swift_bindings/xla_tensor_tf_ops.cc
  swift_bindings/xla_tensor_wrapper.cc)
set_target_properties(x10_c_wrappers PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_link_libraries(x10_c_wrappers PUBLIC
  x10)

add_library(x10_device SHARED
  swift_bindings/Device.swift)
target_link_libraries(x10_device PRIVATE
  x10_c_wrappers)

add_library(x10_tensor SHARED
  ../TensorFlow/Core/BroadcastingPullback.swift

  ../TensorFlow/Operators/Math.swift

  swift_bindings/apis/CrossReplicaSum.swift
  swift_bindings/apis/DataTypes.swift
  swift_bindings/apis/DeviceScope.swift
  swift_bindings/apis/MixedPrecision.swift
  swift_bindings/apis/RawOpsManual.swift

  swift_bindings/TensorFlow/Core/Tensor.swift
  swift_bindings/TensorFlow/Core/Utilities.swift
  swift_bindings/TensorFlow/Optimizers/LARS.swift

  swift_bindings/XLAScalarType.swift
  swift_bindings/XLATensor.swift)
target_sources(x10_tensor PRIVATE
  ../TensorFlow/Core/CopyableToDevice.swift
  ../TensorFlow/Core/DifferentialOperators.swift
  ../TensorFlow/Core/PythonConversion.swift
  ../TensorFlow/Core/ShapedArray.swift
  ../TensorFlow/Core/TensorShape.swift
  ../TensorFlow/Core/Threading.swift

  ../TensorFlow/Layers/Convolutional.swift
  ../TensorFlow/Layers/Core.swift
  ../TensorFlow/Layers/Dense.swift
  ../TensorFlow/Layers/Dropout.swift
  ../TensorFlow/Layers/Embedding.swift
  ../TensorFlow/Layers/Initialization.swift
  ../TensorFlow/Layers/Normalization.swift
  ../TensorFlow/Layers/Pooling.swift
  ../TensorFlow/Layers/Recurrent.swift
  ../TensorFlow/Layers/Sequential.swift
  ../TensorFlow/Layers/Upsampling.swift

  ../TensorFlow/Operators/Basic.swift
  ../TensorFlow/Operators/Comparison.swift
  ../TensorFlow/Operators/LinearAlgebra.swift
  ../TensorFlow/Operators/NN.swift

  ../TensorFlow/Optimizers/MomentumBased.swift
  ../TensorFlow/Optimizers/Optimizer.swift
  ../TensorFlow/Optimizers/SGD.swift

  ../TensorFlow/BackwardsCompatibility.swift
  ../TensorFlow/Context.swift
  ../TensorFlow/Exports.swift
  ../TensorFlow/Freezable.swift
  ../TensorFlow/Initializers.swift
  ../TensorFlow/Layer.swift
  ../TensorFlow/Loss.swift
  ../TensorFlow/StdlibExtensions.swift)
set_target_properties(x10_tensor PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_link_libraries(x10_tensor PUBLIC
  x10_device
  Tensor)
target_link_libraries(x10_tensor PRIVATE
  x10
  x10_c_wrappers)

add_library(x10_optimizers_tensor_visitor_plan SHARED
  swift_bindings/optimizers/TensorVisitorPlan.swift)
set_target_properties(x10_optimizers_tensor_visitor_plan PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_link_libraries(x10_optimizers_tensor_visitor_plan PUBLIC
  x10_tensor)

add_library(x10_optimizers_optimizer SHARED
  swift_bindings/optimizers/Optimizer.swift
  swift_bindings/optimizers/Optimizers.swift)
set_target_properties(x10_optimizers_optimizer PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_link_libraries(x10_optimizers_optimizer PUBLIC
  x10_optimizers_tensor_visitor_plan
  x10_tensor)

add_library(x10_training_loop SHARED
  swift_bindings/training_loop.swift)
set_target_properties(x10_training_loop PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_link_libraries(x10_training_loop PUBLIC
  x10_tensor)

add_executable(ops_test swift_bindings/ops_test.swift)
target_link_libraries(ops_test PRIVATE
  x10_device
  x10_tensor)

add_executable(xla_tensor_test swift_bindings/xla_tensor_test.swift)
target_link_libraries(xla_tensor_test PRIVATE
  x10_device
  x10_tensor)

add_executable(keypathiterable_test swift_bindings/keypathiterable_test.swift)
target_link_libraries(keypathiterable_test PRIVATE
  x10_device
  x10_tensor)

get_swift_host_arch(swift_arch)
get_swift_host_os(swift_os)

install(TARGETS x10_device
  ARCHIVE DESTINATION lib/swift/${swift_os}
  LIBRARY DESTINATION lib/swift/${swift_os}
  RUNTIME DESTINATION bin)
if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
  install(FILES $<TARGET_PROPERTY:x10_device,Swift_MODULE_DIRECTORY>/x10_device.swiftdoc
    DESTINATION lib/swift/${swift_os}/x10_device.swiftmodule
    RENAME ${swift_arch}.swiftdoc)
  install(FILES $<TARGET_PROPERTY:x10_device,Swift_MODULE_DIRECTORY>/x10_device.swiftmodule
    DESTINATION lib/swift/${swift_os}/x10_device.swiftmodule
    RENAME ${swift_arch}.swiftmodule)
else()
  install(FILES
    $<TARGET_PROPERTY:x10_device,Swift_MODULE_DIRECTORY>/x10_device.swiftdoc
    $<TARGET_PROPERTY:x10_device,Swift_MODULE_DIRECTORY>/x10_device.swiftmodule
    DESTINATION lib/swift/${swift_os}/${swift_arch})
endif()

install(TARGETS x10_tensor
  ARCHIVE DESTINATION lib/swift/${swift_os}
  LIBRARY DESTINATION lib/swift/${swift_os}
  RUNTIME DESTINATION bin)
if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
  install(FILES $<TARGET_PROPERTY:x10_tensor,Swift_MODULE_DIRECTORY>/x10_tensor.swiftdoc
    DESTINATION lib/swift/${swift_os}/x10_tensor.swiftmodule
    RENAME ${swift_arch}.swiftdoc)
  install(FILES $<TARGET_PROPERTY:x10_tensor,Swift_MODULE_DIRECTORY>/x10_tensor.swiftmodule
    DESTINATION lib/swift/${swift_os}/x10_tensor.swiftmodule
    RENAME ${swift_arch}.swiftmodule)
else()
  install(FILES
    $<TARGET_PROPERTY:x10_tensor,Swift_MODULE_DIRECTORY>/x10_tensor.swiftdoc
    $<TARGET_PROPERTY:x10_tensor,Swift_MODULE_DIRECTORY>/x10_tensor.swiftmodule
    DESTINATION lib/swift/${swift_os}/${swift_arch})
endif()

install(TARGETS x10_optimizers_tensor_visitor_plan
  ARCHIVE DESTINATION lib/swift/${swift_os}
  LIBRARY DESTINATION lib/swift/${swift_os}
  RUNTIME DESTINATION bin)
if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
  install(FILES $<TARGET_PROPERTY:x10_optimizers_tensor_visitor_plan,Swift_MODULE_DIRECTORY>/x10_optimizers_tensor_visitor_plan.swiftdoc
    DESTINATION lib/swift/${swift_os}/x10_optimizers_tensor_visitor_plan.swiftmodule
    RENAME ${swift_arch}.swiftdoc)
  install(FILES $<TARGET_PROPERTY:x10_optimizers_tensor_visitor_plan,Swift_MODULE_DIRECTORY>/x10_optimizers_tensor_visitor_plan.swiftmodule
    DESTINATION lib/swift/${swift_os}/x10_optimizers_tensor_visitor_plan.swiftmodule
    RENAME ${swift_arch}.swiftmodule)
else()
  install(FILES
    $<TARGET_PROPERTY:x10_optimizers_tensor_visitor_plan,Swift_MODULE_DIRECTORY>/x10_optimizers_tensor_visitor_plan.swiftdoc
    $<TARGET_PROPERTY:x10_optimizers_tensor_visitor_plan,Swift_MODULE_DIRECTORY>/x10_optimizers_tensor_visitor_plan.swiftmodule
    DESTINATION lib/swift/${swift_os}/${swift_arch})
endif()

install(TARGETS x10_optimizers_optimizer
  ARCHIVE DESTINATION lib/swift/${swift_os}
  LIBRARY DESTINATION lib/swift/${swift_os}
  RUNTIME DESTINATION bin)
if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
  install(FILES $<TARGET_PROPERTY:x10_optimizers_optimizer,Swift_MODULE_DIRECTORY>/x10_optimizers_optimizer.swiftdoc
    DESTINATION lib/swift/${swift_os}/x10_optimizers_optimizer.swiftmodule
    RENAME ${swift_arch}.swiftdoc)
  install(FILES $<TARGET_PROPERTY:x10_optimizers_optimizer,Swift_MODULE_DIRECTORY>/x10_optimizers_optimizer.swiftmodule
    DESTINATION lib/swift/${swift_os}/x10_optimizers_optimizer.swiftmodule
    RENAME ${swift_arch}.swiftmodule)
else()
  install(FILES
    $<TARGET_PROPERTY:x10_optimizers_optimizer,Swift_MODULE_DIRECTORY>/x10_optimizers_optimizer.swiftdoc
    $<TARGET_PROPERTY:x10_optimizers_optimizer,Swift_MODULE_DIRECTORY>/x10_optimizers_optimizer.swiftmodule
    DESTINATION lib/swift/${swift_os}/${swift_arch})
endif()

install(TARGETS x10_training_loop
  ARCHIVE DESTINATION lib/swift/${swift_os}
  LIBRARY DESTINATION lib/swift/${swift_os}
  RUNTIME DESTINATION bin)
if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
  install(FILES $<TARGET_PROPERTY:x10_training_loop,Swift_MODULE_DIRECTORY>/x10_training_loop.swiftdoc
    DESTINATION lib/swift/${swift_os}/x10_training_loop.swiftmodule
    RENAME ${swift_arch}.swiftdoc)
  install(FILES $<TARGET_PROPERTY:x10_training_loop,Swift_MODULE_DIRECTORY>/x10_training_loop.swiftmodule
    DESTINATION lib/swift/${swift_os}/x10_training_loop.swiftmodule
    RENAME ${swift_arch}.swiftmodule)
else()
  install(FILES
    $<TARGET_PROPERTY:x10_training_loop,Swift_MODULE_DIRECTORY>/x10_training_loop.swiftdoc
    $<TARGET_PROPERTY:x10_training_loop,Swift_MODULE_DIRECTORY>/x10_training_loop.swiftmodule
    DESTINATION lib/swift/${swift_os}/${swift_arch})
endif()
