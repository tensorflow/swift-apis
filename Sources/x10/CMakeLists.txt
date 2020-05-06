include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(x10_tensor SHARED
  swift_bindings/apis/CrossReplicaSum.swift
  swift_bindings/apis/DeviceScope.swift
  swift_bindings/apis/RawOpsManual.swift

  swift_bindings/Device.swift
  swift_bindings/XLAScalarType.swift
  swift_bindings/XLATensor.swift)
target_sources(x10_tensor PRIVATE
  ../TensorFlow/Bindings/EagerExecution.swift
  ../TensorFlow/Bindings/RawOpsGenerated.swift
  ../TensorFlow/Bindings/RawOpsAugmented.swift
  ../TensorFlow/Bindings/RawOpsDispatching.swift
  ../TensorFlow/Bindings/TFTensorOperation.swift

  ../TensorFlow/Core/BroadcastingPullback.swift
  ../TensorFlow/Core/CopyableToDevice.swift
  ../TensorFlow/Core/DifferentialOperators.swift
  ../TensorFlow/Core/PythonConversion.swift
  ../TensorFlow/Core/ShapedArray.swift
  ../TensorFlow/Core/Execution.swift
  ../TensorFlow/Core/TensorShape.swift
  ../TensorFlow/Core/Threading.swift
  ../TensorFlow/Core/Utilities.swift
  ../TensorFlow/Core/Tensor.swift
  ../TensorFlow/Core/TensorHandle.swift
  ../TensorFlow/Core/DataTypes.swift
  ../TensorFlow/Core/MixedPrecision.swift
  ../TensorFlow/Core/Runtime.swift
  ../TensorFlow/Core/StringTensor.swift
  ../TensorFlow/Core/TensorGroup.swift

  ../TensorFlow/Core/ArrayOps.swift
  ../TensorFlow/Core/LazyTensorContext.swift
  ../TensorFlow/Core/LazyTensorOperation.swift
  ../TensorFlow/Core/LazyTensorOperation.swift
  ../TensorFlow/Core/LazyTensorShapeInference.swift
  ../TensorFlow/Core/LazyTensorTFFunctionBuilder.swift
  ../TensorFlow/Core/LazyTensorTrace.swift
  ../TensorFlow/Core/LazyTensorTraceCache.swift
  ../TensorFlow/Core/Serialization.swift

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
  ../TensorFlow/Operators/Dataset.swift
  ../TensorFlow/Operators/Image.swift
  ../TensorFlow/Operators/LinearAlgebra.swift
  ../TensorFlow/Operators/Math.swift
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
target_compile_definitions(x10_tensor PRIVATE
  USING_X10_BACKEND)
target_link_libraries(x10_tensor PUBLIC
  CTensorFlow
  Tensor)
target_link_libraries(x10_tensor PRIVATE
  CX10
  x10)

add_library(x10_optimizers_tensor_visitor_plan SHARED
  swift_bindings/optimizers/TensorVisitorPlan.swift)
set_target_properties(x10_optimizers_tensor_visitor_plan PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_link_libraries(x10_optimizers_tensor_visitor_plan PUBLIC
  TensorFlow)

add_library(x10_optimizers_optimizer SHARED
  swift_bindings/optimizers/Optimizer.swift
  swift_bindings/optimizers/Optimizers.swift)
set_target_properties(x10_optimizers_optimizer PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_link_libraries(x10_optimizers_optimizer PUBLIC
  x10_optimizers_tensor_visitor_plan
  TensorFlow)

add_library(x10_training_loop SHARED
  swift_bindings/training_loop.swift)
set_target_properties(x10_training_loop PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_link_libraries(x10_training_loop PUBLIC
  TensorFlow)
target_link_libraries(x10_training_loop PRIVATE
  CX10)


_install_target(x10_optimizers_tensor_visitor_plan)
_install_target(x10_optimizers_optimizer)
_install_target(x10_training_loop)
