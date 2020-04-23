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

extension Tensor where Scalar: TensorFlowFloatingPoint {
  /// Computes dropout given a probability.
  // TODO: Remove the underscore once `droppingOut(probability:)` has been removed.
  @differentiable(wrt: self where Scalar: Differentiable)
  fileprivate func _droppingOut(probability: Double) -> Tensor {
    let noise = Tensor(randomUniform: shape)
    let keepMask = noise .>= Scalar(probability)
    let keepProbability = Scalar(1.0 - probability)
    return self * Tensor(copying: Tensor(keepMask), to: device) / Tensor(keepProbability, on: device)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  /// Computes dropout given a probability.
  @available(
    *, deprecated,
    message:
      """
        This API will be removed after Swift for TensorFlow 0.6.
        For dropout, use the `Dropout` layer.
        """
  )
  @differentiable(wrt: self where Scalar: Differentiable)
  public func droppingOut(probability: Double) -> Tensor {
    _droppingOut(probability: probability)
  }
}

/// A dropout layer.
///
/// Dropout consists in randomly setting a fraction of input units to `0` at each update during
/// training time, which helps prevent overfitting.
@frozen
public struct Dropout<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  @noDerivative public let probability: Double

  /// Creates a dropout layer.
  ///
  /// - Parameter probability: The probability of a node dropping out.
  /// - Precondition: probability must be a value between 0 and 1 (inclusive). 
  public init(probability: Double) {
    precondition(
      0...1 ~= probability,
      "Probability must be a value between 0 and 1 (inclusive) but is \(probability)")
    self.probability = probability
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    switch Context.local.learningPhase {
    case .training:
      return input._droppingOut(probability: probability)
    case .inference:
      return input
    }
  }
}

/// An Alpha dropout layer.
///
/// Alpha Dropout is a `Dropout` that keeps mean and variance of inputs to their 
/// original values, in order to ensure the self-normalizing property even after this
/// dropout. Alpha Dropout fits well to Scaled Exponential Linear Units by randomly
/// setting activations to the negative saturation value.
///
/// Source : Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
@frozen
public struct AlphaDropout<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  @noDerivative public let probability: Double

  /// Initializes an `AlphaDropout` layer with a configurable `probability`.
  ///
  /// - Parameter probability: The probability of a node dropping out.
  /// - Precondition: probability must be a value between 0 and 1 (inclusive). 
  public init(probability: Double) {
    precondition(
      0...1 ~= probability,
      "Probability must be a value between 0 and 1 (inclusive) but is \(probability)")
    self.probability = probability
  }

  /// Adds noise to `input` during training, and is a no-op during inference.
  @differentiable
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    switch Context.local.learningPhase {
    case .training:
      let alpha = 1.6732632423543772848170429916717
      let scale = 1.0507009873554804934193349852946
      let alpha_p = -alpha * scale
      let uniform = Tensor<Scalar>(randomUniform: input.shape)
      let noise = uniform .>= Scalar(probability)

      // Get affine transformation params
      let a = pow(((1 - probability) * (1 + probability * pow(alpha_p, 2))), -0.5)
      let b = -a * alpha_p * probability

      // Apply mask
      var x = input * Tensor(noise)
      x = x + Scalar(alpha_p) * (1 - Tensor(noise))

      // Do affine transformation
      return Scalar(a) * x + Scalar(b)
    case .inference:
      return input
    }
  }
}
