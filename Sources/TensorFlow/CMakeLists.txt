add_library(TensorFlow SHARED
  BackwardsCompatibility.swift
  Context.swift
  Exports.swift
  Freezable.swift
  Initializers.swift
  Layer.swift
  Loss.swift
  StdlibExtensions.swift

  Bindings/EagerExecution.swift
  Bindings/RawOpsGenerated.swift
  Bindings/TFTensorOperation.swift
  Bindings/RawOpsAugmented.swift
  Bindings/RawOpsDispatching.swift

  Core/ArrayOps.swift
  Core/CopyableToDevice.swift
  Core/DataTypes.swift
  Core/DifferentialOperators.swift
  Core/Execution.swift
  Core/LazyTensorContext.swift
  Core/LazyTensorOperation.swift
  Core/LazyTensorShapeInference.swift
  Core/LazyTensorTFFunctionBuilder.swift
  Core/LazyTensorTrace.swift
  Core/LazyTensorTraceCache.swift
  Core/BroadcastingPullback.swift
  Core/MixedPrecision.swift
  Core/PythonConversion.swift
  Core/Runtime.swift
  Core/Serialization.swift
  Core/ShapedArray.swift
  Core/StringTensor.swift
  Core/Tensor.swift
  Core/TensorGroup.swift
  Core/TensorHandle.swift
  Core/TensorShape.swift
  Core/Threading.swift
  Core/Utilities.swift
  
  Epochs/Algorithms.swift
  Epochs/Backend.swift
  Epochs/Collatable.swift
  Epochs/NonuniformTrainingEpochs.swift
  Epochs/Sampling.swift
  Epochs/Slices.swift
  Epochs/TrainingEpochs.swift

  Layers/Convolutional.swift
  Layers/Core.swift
  Layers/Dense.swift
  Layers/Dropout.swift
  Layers/Embedding.swift
  Layers/Initialization.swift
  Layers/Morphological.swift
  Layers/Normalization.swift
  Layers/Pooling.swift
  Layers/Recurrent.swift
  Layers/Sequential.swift
  Layers/Upsampling.swift

  Operators/Basic.swift
  Operators/Comparison.swift
  Operators/Dataset.swift
  Operators/Image.swift
  Operators/LinearAlgebra.swift
  Operators/Math.swift
  Operators/NN.swift

  Optimizers/MomentumBased.swift
  Optimizers/Optimizer.swift
  Optimizers/SGD.swift)
target_sources(TensorFlow PRIVATE
  ../x10/swift_bindings/apis/CrossReplicaSum.swift
  ../x10/swift_bindings/apis/DeviceScope.swift
  ../x10/swift_bindings/apis/RawOpsManual.swift

  ../x10/swift_bindings/Device.swift
  ../x10/swift_bindings/XLAScalarType.swift
  ../x10/swift_bindings/XLATensor.swift)
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
