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

/// A batch normalization layer.
///
/// Normalizes the activations of the previous layer at each batch, i.e. applies a transformation
/// that maintains the mean activation close to `0` and the activation standard deviation close to
/// `1`.
///
/// Reference: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal
/// Covariate Shift](https://arxiv.org/abs/1502.03167).
@frozen
public struct BatchNorm<Scalar: TensorFlowFloatingPoint>: Layer {
  /// The feature dimension.
  @noDerivative public let axis: Int
  /// The momentum for the running mean and running variance.
  @noDerivative public let momentum: Scalar
  /// The offset value, also known as beta.
  public var offset: Tensor<Scalar>
  /// The scale value, also known as gamma.
  public var scale: Tensor<Scalar>
  /// The variance epsilon value.
  @noDerivative public let epsilon: Scalar
  /// The running mean.
  @noDerivative public var runningMean: Parameter<Scalar>
  /// The running variance.
  @noDerivative public var runningVariance: Parameter<Scalar>

  /// Creates a batch normalization layer.
  ///
  /// - Parameters:
  ///   - axis: The axis that should not be normalized (typically the feature axis).
  ///   - momentum: The momentum for the moving average.
  ///   - offset: The offset to be added to the normalized tensor.
  ///   - scale: The scale to multiply the normalized tensor by.
  ///   - epsilon: A small scalar added to the denominator to improve numerical stability.
  ///   - runningMean: The running mean.
  ///   - runningVariance: The running variance.
  public init(
    axis: Int,
    momentum: Scalar,
    offset: Tensor<Scalar>,
    scale: Tensor<Scalar>,
    epsilon: Scalar,
    runningMean: Tensor<Scalar>,
    runningVariance: Tensor<Scalar>
  ) {
    precondition(offset.rank == 1, "The offset must have rank 1.")
    precondition(scale.rank == 1, "The scale must have rank 1.")
    precondition(
      offset.shape == scale.shape,
      "The offset and the scale must have same shape.")
    self.axis = axis
    self.momentum = momentum
    self.offset = offset
    self.scale = scale
    self.epsilon = epsilon
    self.runningMean = Parameter(runningMean)
    self.runningVariance = Parameter(runningVariance)
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let positiveAxis = (input.rank + axis) % input.rank
    precondition(
      input.shape[positiveAxis] == offset.shape[0],
      "The number of features of the input and the offset doesn't match.")
    var offset = self.offset
    var scale = self.scale
    if positiveAxis != input.rank - 1 {
      var broadcastShape = TensorShape([Int](repeating: 1, count: input.rank))
      broadcastShape[positiveAxis] = input.shape[positiveAxis]
      offset = offset.reshaped(to: broadcastShape)
      scale = scale.reshaped(to: broadcastShape)
    }
    switch Context.local.learningPhase {
    case .training:
      return doTraining(input, offset: offset, scale: scale, axis: positiveAxis)
    case .inference:
      return doInference(input, offset: offset, scale: scale)
    }
  }

  private func doTraining(
    _ input: Tensor<Scalar>, offset: Tensor<Scalar>, scale: Tensor<Scalar>, axis: Int
  ) -> Tensor<Scalar> {
    var normalizedAxes = Array(0..<input.rank)
    normalizedAxes.remove(at: axis)
    let moments = input.moments(alongAxes: normalizedAxes)
    let decayMomentum = Tensor(1 - momentum, on: input.device)
    let isReducedPrecision = withoutDerivative(at: input) { $0.isReducedPrecision }
    var momentsMean = moments.mean
    var momentsVariance = moments.variance
    if isReducedPrecision {
      momentsMean = momentsMean.toFullPrecision
      momentsVariance = momentsVariance.toFullPrecision
    }
    runningMean.value += (momentsMean - runningMean.value) * decayMomentum
    runningVariance.value += (momentsVariance - runningVariance.value) * decayMomentum
    let eps = withoutDerivative(at: input) { Tensor(epsilon, deviceAndPrecisionLike: $0) }
    let inv = rsqrt(moments.variance + eps) * scale
    return (input - moments.mean) * inv + offset
  }

  private func doInference(
    _ input: Tensor<Scalar>, offset: Tensor<Scalar>, scale: Tensor<Scalar>
  ) -> Tensor<Scalar> {
    let isReducedPrecision = withoutDerivative(at: input) { $0.isReducedPrecision }
    let runningVarianceValue =
      isReducedPrecision ? runningVariance.value.toReducedPrecision : runningVariance.value
    let runningMeanValue =
      isReducedPrecision ? runningMean.value.toReducedPrecision : runningMean.value
    let eps = withoutDerivative(at: input) { Tensor(epsilon, deviceAndPrecisionLike: $0) }
    let inv = rsqrt(runningVarianceValue + eps) * scale
    return (input - runningMeanValue) * inv + offset
  }

  /// Creates a batch normalization layer.
  ///
  /// - Parameters:
  ///   - featureCount: The number of features.
  ///   - axis: The axis that should be normalized (typically the features axis).
  ///   - momentum: The momentum for the moving average.
  ///   - epsilon: A small scalar added to the denominator to improve numerical stability.
  public init(
    featureCount: Int,
    axis: Int = -1,
    momentum: Scalar = 0.99,
    epsilon: Scalar = 0.001
  ) {
    self.init(
      axis: axis,
      momentum: momentum,
      offset: Tensor(zeros: [featureCount]),
      scale: Tensor(ones: [featureCount]),
      epsilon: epsilon,
      runningMean: Tensor(0),
      runningVariance: Tensor(1))
  }
}

/// A layer that applies layer normalization over a mini-batch of inputs.
///
/// Reference: [Layer Normalization](https://arxiv.org/abs/1607.06450).
@frozen
public struct LayerNorm<Scalar: TensorFlowFloatingPoint>: Layer {
  /// The offset value, also known as beta.
  public var offset: Tensor<Scalar>
  /// The scale value, also known as gamma.
  public var scale: Tensor<Scalar>
  /// The axis.
  @noDerivative public let axis: Int
  /// The variance epsilon value.
  @noDerivative public let epsilon: Scalar

  /// Creates a layer normalization layer.
  public init(
    offset: Tensor<Scalar>,
    scale: Tensor<Scalar>,
    axis: Int,
    epsilon: Scalar
  ) {
    precondition(offset.rank == 1, "The offset must have rank 1.")
    precondition(scale.rank == 1, "The scale must have rank 1.")
    precondition(
      offset.shape == scale.shape,
      "The offset and the scale must have same shape.")
    self.offset = offset
    self.scale = scale
    self.axis = axis
    self.epsilon = epsilon
  }

  /// Creates a layer normalization layer.
  ///
  /// - Parameters:
  ///   - featureCount: The number of features.
  ///   - axis: The axis that should be normalized.
  ///   - epsilon: The small scalar added to variance.
  public init(
    featureCount: Int,
    axis: Int,
    epsilon: Scalar = 0.001
  ) {
    self.init(
      offset: Tensor(zeros: [featureCount]),
      scale: Tensor(ones: [featureCount]),
      axis: axis,
      epsilon: epsilon)
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let epsilon = Tensor(self.epsilon, on: input.device)
    let positiveAxis = (input.rank + axis) % input.rank
    precondition(
      input.shape[positiveAxis] == offset.shape[0],
      "The number of features of the input and the offset doesn't match.")
    var broadcastShape = TensorShape(Array(repeating: 1, count: input.rank))
    broadcastShape[positiveAxis] = input.shape[positiveAxis]
    let offset = self.offset.reshaped(to: broadcastShape)
    let scale = self.scale.reshaped(to: broadcastShape)
    let moments = input.moments(alongAxes: positiveAxis)
    let inv = rsqrt(moments.variance + epsilon) * scale
    return (input - moments.mean) * inv + offset
  }
}
