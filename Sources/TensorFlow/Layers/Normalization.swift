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

/// Returns normalized `input`.
///
/// - Parameters:
///   - input: The tensor to be normalized.
///   - mean: The mean tensor.
///   - variance: The variance tensor.
///   - offset: The tensor to be added to normalized tensor.
///   - scale: The tensor to be applied to normalized tensor.
///   - varianceEpsilon: The small number to avoid dividing by 0.
@differentiable(wrt: (input, mean, variance, offset, scale))
private func normalize<Scalar: TensorFlowFloatingPoint>(
  _ input: Tensor<Scalar>,
  mean: Tensor<Scalar>,
  variance: Tensor<Scalar>,
  offset: Tensor<Scalar>,
  scale: Tensor<Scalar>,
  varianceEpsilon: Tensor<Scalar>
) -> Tensor<Scalar> {
  let inv = scale * rsqrt(variance + varianceEpsilon)
  return input * inv + (offset - mean * inv)
}

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
    return normalize(
      input,
      mean: moments.mean, variance: moments.variance,
      offset: offset, scale: scale,
      varianceEpsilon: eps)
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
    return normalize(
      input,
      mean: runningMeanValue, variance: runningVarianceValue,
      offset: offset, scale: scale,
      varianceEpsilon: eps)
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

/// A layer that applies group normalization over a mini-batch of inputs.
///
/// Reference: [Group Normalization](https://arxiv.org/abs/1803.08494).
@frozen
public struct GroupNorm<Scalar: TensorFlowFloatingPoint>: Layer {
  /// The offset value, also known as beta.
  public var offset: Tensor<Scalar>
  /// The scale value, also known as gamma.
  public var scale: Tensor<Scalar>
  /// The number of groups.
  @noDerivative public let groups: Int
  /// The axis.
  @noDerivative public let axis: Int
  /// The variance epsilon value.
  @noDerivative public let epsilon: Scalar

  /// Creates a group normalization layer.
  public init(
    offset: Tensor<Scalar>,
    scale: Tensor<Scalar>,
    groups: Int,
    axis: Int,
    epsilon: Scalar
  ) {
    precondition(axis != 0, "The axis cannot be batch axis.")
    precondition(offset.rank == 1, "The offset must have rank 1.")
    precondition(scale.rank == 1, "The scale must have rank 1.")
    precondition(
      offset.shape == scale.shape,
      "The offset and the scale must have same shape.")
    self.offset = offset
    self.scale = scale
    self.groups = groups
    self.axis = axis
    self.epsilon = epsilon
  }

  /// Creates a group normalization layer.
  ///
  /// - Parameters:
  ///   - featureCount: The number of features.
  ///   - groups: The number of groups.
  ///   - axis: The axis that should be normalized.
  ///   - epsilon: The small scalar added to variance.
  public init(
    featureCount: Int,
    groups: Int,
    axis: Int = -1,
    epsilon: Scalar = 1e-3
  ) {
    precondition(
      featureCount.isMultiple(of: groups),
      "The feature count must be divisible by groups.")
    self.init(
      offset: Tensor(zeros: [featureCount]),
      scale: Tensor(ones: [featureCount]),
      groups: groups,
      axis: axis,
      epsilon: epsilon
    )
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let positiveAxis = (input.rank + axis) % input.rank
    precondition(positiveAxis != 0, "The axis cannot be batch axis.")
    var offset = self.offset
    var scale = self.scale
    var broadcastShape = TensorShape([Int](repeating: 1, count: input.rank + 1))
    broadcastShape[positiveAxis] = groups
    broadcastShape[positiveAxis + 1] = input.shape[positiveAxis] / groups
    offset = offset.reshaped(to: broadcastShape)
    scale = scale.reshaped(to: broadcastShape)

    var groupShape = input.shape
    groupShape[positiveAxis] /= groups
    groupShape.insert(groups, at: positiveAxis)
    let grouped = input.reshaped(to: groupShape)
    var normalizedAxes = Array(1..<grouped.rank)
    normalizedAxes.remove(at: positiveAxis - 1)
    let moments = grouped.moments(alongAxes: normalizedAxes)
    let eps = Tensor(epsilon)
    let normalized = normalize(
      grouped,
      mean: moments.mean, variance: moments.variance,
      offset: offset, scale: scale,
      varianceEpsilon: eps)
    return normalized.reshaped(to: input.shape)
  }
}

/// A layer that applies instance normalization over a mini-batch of inputs.
///
/// Reference: [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022).
@frozen
public struct InstanceNorm<Scalar: TensorFlowFloatingPoint>: Layer {
  /// Internal group normalization.
  var groupNorm: GroupNorm<Scalar>

  /// The offset value, also known as beta.
  public var offset: Tensor<Scalar> {
    _read { yield groupNorm.offset }
    _modify { yield &groupNorm.offset }
  }

  /// The scale value, also known as gamma.
  public var scale: Tensor<Scalar> {
    _read { yield groupNorm.scale }
    _modify { yield &groupNorm.scale }
  }

  /// The axis.
  public var axis: Int { groupNorm.axis }
  /// The variance epsilon value.
  public var epsilon: Scalar { groupNorm.epsilon }

  /// Creates a instance normalization layer.
  public init(
    offset: Tensor<Scalar>,
    scale: Tensor<Scalar>,
    axis: Int,
    epsilon: Scalar
  ) {
    groupNorm = GroupNorm(
      offset: offset,
      scale: scale,
      groups: offset.shape[0],
      axis: axis,
      epsilon: epsilon)
  }

  /// Creates a instance normalization layer.
  ///
  /// - Parameters:
  ///   - featureCount: The number of features.
  ///   - axis: The axis that should be normalized.
  ///   - epsilon: The small scalar added to variance.
  public init(
    featureCount: Int,
    axis: Int = -1,
    epsilon: Scalar = 1e-3
  ) {
    groupNorm = GroupNorm(
      featureCount: featureCount,
      groups: featureCount,
      axis: axis,
      epsilon: epsilon)
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    groupNorm(input)
  }
}
