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

/// A densely-connected neural network layer.
///
/// `Dense` implements the operation `activation(matmul(input, weight) + bias)`, where `weight` is
/// a weight matrix, `bias` is a bias vector, and `activation` is an element-wise activation
/// function.
///
/// This layer also supports 3-D weight tensors with 2-D bias matrices. In this case the first
/// dimension of both is treated as the batch size that is aligned with the first dimension of
/// `input` and the batch variant of the `matmul(_:_:)` operation is used, thus using a different
/// weight and bias for each element in input batch.
@frozen
public struct Dense<Scalar: TensorFlowFloatingPoint>: Layer {
  /// The weight matrix.
  public var weight: Tensor<Scalar>
  /// The bias vector.
  public var bias: Tensor<Scalar>
  /// The element-wise activation function.
  @noDerivative public let activation: Activation
  /// Indicates whether this is a batched dense layer.
  @noDerivative internal let batched: Bool
  /// Workaround optionals not being handled by AD
  @noDerivative private let useBias: Bool

  /// The element-wise activation function type.
  public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

  /// Creates an instance from the given weight, optional bias, and activation function.
  ///
  /// - Note: currently, `weight` is the only differentiability parameter. `bias` can be made a
  ///   differentiability parameter after `Optional` conditionally conforms to `Differentiable`:
  ///   TF-499.
  @differentiable(wrt: weight)
  public init(
    weight: Tensor<Scalar>,
    bias: Tensor<Scalar>? = nil,
    activation: @escaping Activation
  ) {
    precondition(weight.rank <= 3, "The rank of the 'weight' tensor must be less than 4.")
    precondition(
      bias == nil || bias!.rank <= 2, "The rank of the 'bias' tensor must be less than 3.")
    self.weight = weight
    self.bias = bias ?? .zero
    self.activation = activation
    self.batched = weight.rank == 3
    useBias = (bias != nil)
  }

  // TODO(TF-433): Remove custom derivative after `try_apply` differentiation is supported.
  @derivative(of: init, wrt: weight)
  @usableFromInline
  static func vjpInit(
    weight: Tensor<Scalar>,
    bias: Tensor<Scalar>? = nil,
    activation: @escaping Activation
  ) -> (value: Self, pullback: (TangentVector) -> Tensor<Scalar>) {
    let value = Dense(weight: weight, bias: bias, activation: activation)
    return (value, { v in v.weight })
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    if batched {
      let hidden = matmul(input.expandingShape(at: 1), weight).squeezingShape(at: 1)
      return activation(useBias ? hidden + bias : hidden)
    }
    return activation(useBias ? (matmul(input, weight) + bias) : matmul(input, weight))
  }

  @derivative(of: forward, wrt: self)
  @usableFromInline
  func _jvpForward(_ input: Tensor<Scalar>) -> (
    value: Tensor<Scalar>,
    differential: (TangentVector) -> Tensor<Scalar>
  ) {
    fatalError("Forward-mode derivative is not yet implemented")
  }

  @derivative(of: forward)
  @usableFromInline
  func _jvpForward(_ input: Tensor<Scalar>) -> (
    value: Tensor<Scalar>,
    differential: (TangentVector, Tensor<Scalar>) -> Tensor<Scalar>
  ) {
    fatalError("Forward-mode derivative is not yet implemented")
  }
}

extension Dense {
  /// Creates a `Dense` layer with the specified input size, output size, and element-wise
  /// activation function. The weight matrix is created with shape `[inputSize, outputSize]` and
  /// the bias vector is created with shape `[outputSize]`.
  ///
  /// - Parameters:
  ///   - inputSize: The dimensionality of the input space.
  ///   - outputSize: The dimensionality of the output space.
  ///   - activation: The activation function to use. The default value is `identity(_:)`.
  ///   - weightInitializer: Initializer to use for `weight`.
  ///   - biasInitializer: Initializer to use for `bias`.
  public init(
    inputSize: Int,
    outputSize: Int,
    activation: @escaping Activation = identity,
    useBias: Bool = true,
    weightInitializer: ParameterInitializer<Scalar> = glorotUniform(),
    biasInitializer: ParameterInitializer<Scalar> = zeros()
  ) {
    self.init(
      weight: weightInitializer([inputSize, outputSize]),
      bias: useBias ? biasInitializer([outputSize]) : nil,
      activation: activation)
  }
}
