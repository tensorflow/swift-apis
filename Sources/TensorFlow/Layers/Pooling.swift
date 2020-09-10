// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import _Differentiation

/// A max pooling layer for temporal data.
@frozen
public struct MaxPool1D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias TangentVector = EmptyTangentVector

  /// The size of the sliding reduction window for pooling.
  @noDerivative public let poolSize: Int
  /// The stride of the sliding window for temporal dimension.
  @noDerivative public let stride: Int
  /// The padding algorithm for pooling.
  @noDerivative public let padding: Padding

  /// Creates a max pooling layer.
  ///
  /// - Parameters:
  ///   - poolSize: The size of the sliding reduction window for pooling.
  ///   - stride: The stride of the sliding window for temporal dimension.
  ///   - padding: The padding algorithm for pooling.
  public init(poolSize: Int, stride: Int, padding: Padding) {
    precondition(poolSize > 0, "The pooling window size must be greater than 0.")
    precondition(stride > 0, "The stride must be greater than 0.")
    self.poolSize = poolSize
    self.stride = stride
    self.padding = padding
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    maxPool2D(
      input.expandingShape(at: 1),
      filterSize: (1, 1, poolSize, 1),
      strides: (1, 1, stride, 1),
      padding: padding
    ).squeezingShape(at: 1)
  }
}

/// A max pooling layer for spatial data.
@frozen
public struct MaxPool2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias TangentVector = EmptyTangentVector

  /// The size of the sliding reduction window for pooling.
  @noDerivative public let poolSize: (Int, Int, Int, Int)
  /// The strides of the sliding window for each dimension of a 4-D input.
  /// Strides in non-spatial dimensions must be `1`.
  @noDerivative public let strides: (Int, Int, Int, Int)
  /// The padding algorithm for pooling.
  @noDerivative public let padding: Padding

  /// Creates a max pooling layer.
  public init(poolSize: (Int, Int, Int, Int), strides: (Int, Int, Int, Int), padding: Padding) {
    precondition(
      poolSize.0 > 0 && poolSize.1 > 0 && poolSize.2 > 0 && poolSize.3 > 0,
      "Pooling window sizes must be greater than 0.")
    precondition(
      strides.0 > 0 && strides.1 > 0 && strides.2 > 0 && strides.3 > 0,
      "Strides must be greater than 0.")
    self.poolSize = poolSize
    self.strides = strides
    self.padding = padding
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    maxPool2D(input, filterSize: poolSize, strides: strides, padding: padding)
  }
}

extension MaxPool2D {
  /// Creates a max pooling layer.
  ///
  /// - Parameters:
  ///   - poolSize: Vertical and horizontal factors by which to downscale.
  ///   - strides: The strides.
  ///   - padding: The padding.
  public init(poolSize: (Int, Int), strides: (Int, Int), padding: Padding = .valid) {
    self.init(
      poolSize: (1, poolSize.0, poolSize.1, 1),
      strides: (1, strides.0, strides.1, 1),
      padding: padding)
  }
}

/// A max pooling layer for spatial or spatio-temporal data.
@frozen
public struct MaxPool3D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias TangentVector = EmptyTangentVector

  /// The size of the sliding reduction window for pooling.
  @noDerivative public let poolSize: (Int, Int, Int, Int, Int)
  /// The strides of the sliding window for each dimension of a 5-D input.
  /// Strides in non-spatial dimensions must be `1`.
  @noDerivative public let strides: (Int, Int, Int, Int, Int)
  /// The padding algorithm for pooling.
  @noDerivative public let padding: Padding

  /// Creates a max pooling layer.
  public init(
    poolSize: (Int, Int, Int, Int, Int),
    strides: (Int, Int, Int, Int, Int),
    padding: Padding
  ) {
    precondition(
      poolSize.0 > 0 && poolSize.1 > 0 && poolSize.2 > 0 && poolSize.3 > 0 && poolSize.4 > 0,
      "Pooling window sizes must be greater than 0."
    )
    precondition(
      strides.0 > 0 && strides.1 > 0 && strides.2 > 0 && strides.3 > 0 && strides.4 > 0,
      "Strides must be greater than 0."
    )
    self.poolSize = poolSize
    self.strides = strides
    self.padding = padding
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    maxPool3D(input, filterSize: poolSize, strides: strides, padding: padding)
  }
}

extension MaxPool3D {
  /// Creates a max pooling layer.
  ///
  /// - Parameters:
  ///   - poolSize: Vertical and horizontal factors by which to downscale.
  ///   - strides: The strides.
  ///   - padding: The padding.
  public init(poolSize: (Int, Int, Int), strides: (Int, Int, Int), padding: Padding = .valid) {
    self.init(
      poolSize: (1, poolSize.0, poolSize.1, poolSize.2, 1),
      strides: (1, strides.0, strides.1, strides.2, 1),
      padding: padding)
  }
}

extension MaxPool3D {
  /// Creates a max pooling layer with the specified pooling window size and stride. All pooling
  /// sizes and strides are the same.
  public init(poolSize: Int, stride: Int, padding: Padding = .valid) {
    self.init(
      poolSize: (poolSize, poolSize, poolSize),
      strides: (stride, stride, stride),
      padding: padding)
  }
}

/// An average pooling layer for temporal data.
@frozen
public struct AvgPool1D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias TangentVector = EmptyTangentVector

  /// The size of the sliding reduction window for pooling.
  @noDerivative public let poolSize: Int
  /// The stride of the sliding window for temporal dimension.
  @noDerivative public let stride: Int
  /// The padding algorithm for pooling.
  @noDerivative public let padding: Padding

  /// Creates an average pooling layer.
  ///
  /// - Parameters:
  ///   - poolSize: The size of the sliding reduction window for pooling.
  ///   - stride: The stride of the sliding window for temporal dimension.
  ///   - padding: The padding algorithm for pooling.
  public init(poolSize: Int, stride: Int, padding: Padding) {
    precondition(poolSize > 0, "The pooling window size must be greater than 0.")
    precondition(stride > 0, "The stride must be greater than 0.")
    self.poolSize = poolSize
    self.stride = stride
    self.padding = padding
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    avgPool2D(
      input.expandingShape(at: 1),
      filterSize: (1, 1, poolSize, 1),
      strides: (1, 1, stride, 1),
      padding: padding
    ).squeezingShape(at: 1)
  }
}

/// An average pooling layer for spatial data.
@frozen
public struct AvgPool2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias TangentVector = EmptyTangentVector

  /// The size of the sliding reduction window for pooling.
  @noDerivative public let poolSize: (Int, Int, Int, Int)
  /// The strides of the sliding window for each dimension of a 4-D input.
  /// Strides in non-spatial dimensions must be `1`.
  @noDerivative public let strides: (Int, Int, Int, Int)
  /// The padding algorithm for pooling.
  @noDerivative public let padding: Padding

  /// Creates an average pooling layer.
  public init(poolSize: (Int, Int, Int, Int), strides: (Int, Int, Int, Int), padding: Padding) {
    precondition(
      poolSize.0 > 0 && poolSize.1 > 0 && poolSize.2 > 0 && poolSize.3 > 0,
      "Pooling window sizes must be greater than 0.")
    precondition(
      strides.0 > 0 && strides.1 > 0 && strides.2 > 0 && strides.3 > 0,
      "Strides must be greater than 0.")
    self.poolSize = poolSize
    self.strides = strides
    self.padding = padding
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    avgPool2D(input, filterSize: poolSize, strides: strides, padding: padding)
  }
}

extension AvgPool2D {
  /// Creates an average pooling layer.
  ///
  /// - Parameters:
  ///   - poolSize: Vertical and horizontal factors by which to downscale.
  ///   - strides: The strides.
  ///   - padding: The padding.
  public init(poolSize: (Int, Int), strides: (Int, Int), padding: Padding = .valid) {
    self.init(
      poolSize: (1, poolSize.0, poolSize.1, 1),
      strides: (1, strides.0, strides.1, 1),
      padding: padding)
  }
}

/// An average pooling layer for spatial or spatio-temporal data.
@frozen
public struct AvgPool3D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias TangentVector = EmptyTangentVector

  /// The size of the sliding reduction window for pooling.
  @noDerivative public let poolSize: (Int, Int, Int, Int, Int)
  /// The strides of the sliding window for each dimension of a 5-D input.
  /// Strides in non-spatial dimensions must be `1`.
  @noDerivative public let strides: (Int, Int, Int, Int, Int)
  /// The padding algorithm for pooling.
  @noDerivative public let padding: Padding

  /// Creates an average pooling layer.
  public init(
    poolSize: (Int, Int, Int, Int, Int),
    strides: (Int, Int, Int, Int, Int),
    padding: Padding
  ) {
    precondition(
      poolSize.0 > 0 && poolSize.1 > 0 && poolSize.2 > 0 && poolSize.3 > 0 && poolSize.4 > 0,
      "Pooling window sizes must be greater than 0."
    )
    precondition(
      strides.0 > 0 && strides.1 > 0 && strides.2 > 0 && strides.3 > 0 && strides.4 > 0,
      "Strides must be greater than 0."
    )
    self.poolSize = poolSize
    self.strides = strides
    self.padding = padding
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    avgPool3D(input, filterSize: poolSize, strides: strides, padding: padding)
  }
}

extension AvgPool3D {
  /// Creates an average pooling layer.
  ///
  /// - Parameters:
  ///   - poolSize: Vertical and horizontal factors by which to downscale.
  ///   - strides: The strides.
  ///   - padding: The padding.
  public init(poolSize: (Int, Int, Int), strides: (Int, Int, Int), padding: Padding = .valid) {
    self.init(
      poolSize: (1, poolSize.0, poolSize.1, poolSize.2, 1),
      strides: (1, strides.0, strides.1, strides.2, 1),
      padding: padding)
  }
}

extension AvgPool3D {
  /// Creates an average pooling layer with the specified pooling window size and stride. All
  /// pooling sizes and strides are the same.
  public init(poolSize: Int, strides: Int, padding: Padding = .valid) {
    self.init(
      poolSize: (poolSize, poolSize, poolSize),
      strides: (strides, strides, strides),
      padding: padding)
  }
}

/// A global average pooling layer for temporal data.
@frozen
public struct GlobalAvgPool1D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias TangentVector = EmptyTangentVector

  /// Creates a global average pooling layer.
  public init() {}

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    precondition(input.rank == 3, "The rank of the input must be 3.")
    return input.mean(squeezingAxes: 1)
  }

  // Note: this custom JVP function exists as a workaround for forward-mode differentiation issues.
  // Remove it when forward-mode differentiation issues (SR-13530) are fixed.
  @derivative(of: forward)
  @usableFromInline
  func _jvpForward(_ input: Tensor<Scalar>) -> (
    value: Tensor<Scalar>,
    differential: (TangentVector, Tensor<Scalar>) -> Tensor<Scalar>
  ) {
    fatalError("Forward-mode derivative is not yet implemented")
  }
}

/// A global average pooling layer for spatial data.
@frozen
public struct GlobalAvgPool2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias TangentVector = EmptyTangentVector

  /// Creates a global average pooling layer.
  public init() {}

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    precondition(input.rank == 4, "The rank of the input must be 4.")
    return input.mean(squeezingAxes: [1, 2])
  }

  // Note: this custom JVP function exists as a workaround for forward-mode differentiation issues.
  // Remove it when forward-mode differentiation issues (SR-13530) are fixed.
  @derivative(of: forward)
  @usableFromInline
  func _jvpForward(_ input: Tensor<Scalar>) -> (
    value: Tensor<Scalar>,
    differential: (TangentVector, Tensor<Scalar>) -> Tensor<Scalar>
  ) {
    fatalError("Forward-mode derivative is not yet implemented")
  }
}

/// A global average pooling layer for spatial and spatio-temporal data.
@frozen
public struct GlobalAvgPool3D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias TangentVector = EmptyTangentVector

  /// Creates a global average pooling layer.
  public init() {}

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    precondition(input.rank == 5, "The rank of the input must be 5.")
    return input.mean(squeezingAxes: [1, 2, 3])
  }

  // Note: this custom JVP function exists as a workaround for forward-mode differentiation issues.
  // Remove it when forward-mode differentiation issues (SR-13530) are fixed.
  @derivative(of: forward)
  @usableFromInline
  func _jvpForward(_ input: Tensor<Scalar>) -> (
    value: Tensor<Scalar>,
    differential: (TangentVector, Tensor<Scalar>) -> Tensor<Scalar>
  ) {
    fatalError("Forward-mode derivative is not yet implemented")
  }
}

/// A global max pooling layer for temporal data.
@frozen
public struct GlobalMaxPool1D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias TangentVector = EmptyTangentVector

  /// Creates a global max pooling layer.
  public init() {}

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameters:
  ///   - input: The input to the layer.
  ///   - context: The contextual information for the layer application, e.g. the current learning
  ///     phase.
  /// - Returns: The output.
  @differentiable
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    precondition(input.rank == 3, "The rank of the input must be 3.")
    return input.max(squeezingAxes: 1)
  }

  // Note: this custom JVP function exists as a workaround for forward-mode differentiation issues.
  // Remove it when forward-mode differentiation issues (SR-13530) are fixed.
  @derivative(of: forward)
  @usableFromInline
  func _jvpForward(_ input: Tensor<Scalar>) -> (
    value: Tensor<Scalar>,
    differential: (TangentVector, Tensor<Scalar>) -> Tensor<Scalar>
  ) {
    fatalError("Forward-mode derivative is not yet implemented")
  }
}

/// A global max pooling layer for spatial data.
@frozen
public struct GlobalMaxPool2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias TangentVector = EmptyTangentVector

  /// Creates a global max pooling layer.
  public init() {}

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    precondition(input.rank == 4, "The rank of the input must be 4.")
    return input.max(squeezingAxes: [1, 2])
  }

  // Note: this custom JVP function exists as a workaround for forward-mode differentiation issues.
  // Remove it when forward-mode differentiation issues (SR-13530) are fixed.
  @derivative(of: forward)
  @usableFromInline
  func _jvpForward(_ input: Tensor<Scalar>) -> (
    value: Tensor<Scalar>,
    differential: (TangentVector, Tensor<Scalar>) -> Tensor<Scalar>
  ) {
    fatalError("Forward-mode derivative is not yet implemented")
  }
}

/// A global max pooling layer for spatial and spatio-temporal data.
@frozen
public struct GlobalMaxPool3D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias TangentVector = EmptyTangentVector

  /// Creates a global max pooling layer.
  public init() {}

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    precondition(input.rank == 5, "The rank of the input must be 5.")
    return input.max(squeezingAxes: [1, 2, 3])
  }

  // Note: this custom JVP function exists as a workaround for forward-mode differentiation issues.
  // Remove it when forward-mode differentiation issues (SR-13530) are fixed.
  @derivative(of: forward)
  @usableFromInline
  func _jvpForward(_ input: Tensor<Scalar>) -> (
    value: Tensor<Scalar>,
    differential: (TangentVector, Tensor<Scalar>) -> Tensor<Scalar>
  ) {
    fatalError("Forward-mode derivative is not yet implemented")
  }
}

/// A fractional max pooling layer for spatial data.
/// Note: `FractionalMaxPool` does not have an XLA implementation, and thus may have performance implications.
@frozen
public struct FractionalMaxPool2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias TangentVector = EmptyTangentVector

  /// Pooling ratios for each dimension of input of shape (batch, height, width, channels).
  /// Currently pooling in only height and width is supported.
  @noDerivative public let poolingRatio: (Double, Double, Double, Double)
  /// Determines whether pooling sequence is generated by pseudorandom fashion.
  @noDerivative public let pseudoRandom: Bool
  /// Determines whether values at the boundary of adjacent pooling cells are used by both cells
  @noDerivative public let overlapping: Bool
  /// Determines whether a fixed pooling region will be
  /// used when iterating over a FractionalMaxPool2D node in the computation graph.
  @noDerivative public let deterministic: Bool
  /// Seed for the random number generator
  @noDerivative public let seed: Int64
  /// A second seed to avoid seed collision
  @noDerivative public let seed2: Int64
  /// Initializes a `FractionalMaxPool` layer with configurable `poolingRatio`.
  public init(
    poolingRatio: (Double, Double, Double, Double), pseudoRandom: Bool = false,
    overlapping: Bool = false, deterministic: Bool = false, seed: Int64 = 0, seed2: Int64 = 0
  ) {
    precondition(
      poolingRatio.0 == 1.0 && poolingRatio.3 == 1.0,
      "Pooling on batch and channels dimensions not supported.")
    precondition(
      poolingRatio.1 >= 1.0 && poolingRatio.2 >= 1.0,
      "Pooling ratio for height and width dimensions must be at least 1.0")
    self.poolingRatio = poolingRatio
    self.pseudoRandom = pseudoRandom
    self.overlapping = overlapping
    self.deterministic = deterministic
    self.seed = seed
    self.seed2 = seed2
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    fractionalMaxPool2D(
      input,
      poolingRatio: poolingRatio,
      pseudoRandom: pseudoRandom,
      overlapping: overlapping,
      deterministic: deterministic,
      seed: seed,
      seed2: seed2)
  }
}

extension FractionalMaxPool2D {
  /// Creates a fractional max pooling layer.
  ///
  /// - Parameters:
  ///   - poolingRatio: Pooling ratio for height and width dimensions of input.
  ///   - pseudoRandom: Determines wheter the pooling sequence is generated
  ///     in a pseudorandom fashion.
  ///   - overlapping: Determines whether values at the boundary of adjacent
  ///     pooling cells are used by both cells.
  ///   - deterministic: Determines whether a fixed pooling region will be
  ///     used when iterating over a FractionalMaxPool2D node in the computation graph.
  ///   - seed: A seed for random number generator.
  ///   - seed2: A second seed to avoid seed collision.
  public init(
    poolingRatio: (Double, Double), pseudoRandom: Bool = false,
    overlapping: Bool = false, deterministic: Bool = false, seed: Int64 = 0, seed2: Int64 = 0
  ) {
    self.init(
      poolingRatio: (1.0, poolingRatio.0, poolingRatio.1, 1.0),
      pseudoRandom: pseudoRandom,
      overlapping: overlapping,
      deterministic: deterministic,
      seed: seed,
      seed2: seed2)
  }
}
