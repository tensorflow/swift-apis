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

/// A flatten layer.
///
/// A flatten layer flattens the input when applied without affecting the batch size.
@frozen
public struct Flatten<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias TangentVector = EmptyTangentVector

  /// Creates a flatten layer.
  public init() {}

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let batchSize = input.shape[0]
    let remaining = input.shape[1..<input.rank].contiguousSize
    return input.reshaped(to: [batchSize, remaining])
  }
}

/// A reshape layer.
@frozen
public struct Reshape<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  public typealias TangentVector = EmptyTangentVector

  /// The target shape.
  @noDerivative public var shape: Tensor<Int32>

  // TF-331 workaround:
  @usableFromInline
  internal var _nontrivial = Tensor<Float>(0)

  /// Creates a reshape layer.
  ///
  /// - Parameter shape: The target shape, represented by a tensor.
  public init(shape: Tensor<Int32>) {
    self.shape = shape
  }

  /// Creates a reshape layer.
  ///
  /// - Parameter shape: The target shape.
  public init(_ shape: TensorShape) {
    self.init(shape: Tensor(shape.dimensions.map(Int32.init)))
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    return input.reshaped(toShape: shape)
  }
}

/// A layer that encloses a custom differentiable function.
public struct Function<Input: Differentiable, Output: Differentiable>: ParameterlessLayer {
  public typealias TangentVector = EmptyTangentVector
  public typealias Body = @differentiable (Input) -> Output

  @noDerivative public let body: Body

  public init(_ body: @escaping Body) {
    self.body = body
  }

  @differentiable
  public func callAsFunction(_ input: Input) -> Output {
    body(input)
  }

  // Note: this custom JVP function exists as a workaround for forward-mode differentiation issues.
  // Remove it when forward-mode differentiation issues are fixed.
  @derivative(of: callAsFunction, wrt: self)
  @usableFromInline
  func _jvpCallAsFunction(_ input: Input) -> (
    value: Output,
    differential: (TangentVector) -> Output.TangentVector
  ) {
    fatalError("Forward-mode derivative is not yet implemented")
  }

  // Note: this custom JVP function exists as a workaround for forward-mode differentiation issues.
  // Remove it when forward-mode differentiation issues are fixed.
  @derivative(of: callAsFunction)
  @usableFromInline
  func _jvpCallAsFunction(_ input: Input) -> (
    value: Output,
    differential: (TangentVector, Input.TangentVector) -> Output.TangentVector
  ) {
    fatalError("Forward-mode derivative is not yet implemented")
  }
}
