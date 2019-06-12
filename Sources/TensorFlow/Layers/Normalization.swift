// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
    @noDerivative public let momentum: Tensor<Scalar>
    /// The offset value, also known as beta.
    public var offset: Tensor<Scalar>
    /// The scale value, also known as gamma.
    public var scale: Tensor<Scalar>
    /// The variance epsilon value.
    @noDerivative public let epsilon: Tensor<Scalar>
    /// The running mean.
    @noDerivative public let runningMean: Parameter<Scalar>
    /// The running variance.
    @noDerivative public let runningVariance: Parameter<Scalar>

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
        momentum: Tensor<Scalar>,
        offset: Tensor<Scalar>,
        scale: Tensor<Scalar>,
        epsilon: Tensor<Scalar>,
        runningMean: Tensor<Scalar>,
        runningVariance: Tensor<Scalar>
    ) {
        self.axis = axis
        self.momentum = momentum
        self.offset = offset
        self.scale = scale
        self.epsilon = epsilon
        self.runningMean = Parameter(runningMean)
        self.runningVariance = Parameter(runningVariance)
    }

    @differentiable
    private func applyingTraining(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        let positiveAxis = (input.rank + axis) % input.rank
        var normalizedAxes = Array(0..<input.rank)
        normalizedAxes.remove(at: positiveAxis)
        let mean = input.mean(alongAxes: normalizedAxes)
        let variance = input.variance(alongAxes: normalizedAxes)
        runningMean.value += (mean - runningMean.value) * (1 - momentum)
        runningVariance.value += (
            variance - runningVariance.value) * (1 - momentum)
        let inv = rsqrt(variance + epsilon) * scale
        return (input - mean) * inv + offset
    }

    @differentiable
    private func applyingInference(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        let inv = rsqrt(runningVariance.value + epsilon) * scale
        return (input - runningMean.value) * inv + offset
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable(vjp: _vjpApplied(to:))
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        switch Context.local.learningPhase {
        case .training:
            return applyingTraining(to: input)
        case .inference:
            return applyingInference(to: input)
        }
    }

    @usableFromInline
    func _vjpApplied(to input: Tensor<Scalar>) ->
        (Tensor<Scalar>, (Tensor<Scalar>) ->
            (BatchNorm<Scalar>.TangentVector, Tensor<Scalar>)) {
        switch Context.local.learningPhase {
        case .training:
            return valueWithPullback(at: input) {
                $0.applyingTraining(to: $1)
            }
        case .inference:
            return valueWithPullback(at: input) {
                $0.applyingInference(to: $1)
            }
        }
    }

    /// Creates a batch normalization layer.
    ///
    /// - Parameters:
    ///   - featureCount: The number of features.
    ///   - axis: The axis that should be normalized (typically the features axis).
    ///   - momentum: The momentum for the moving average.
    ///   - epsilon: A small scalar added to the denominator to improve numerical stability.
    public init(featureCount: Int,
                axis: Int = -1,
                momentum: Tensor<Scalar> = Tensor(0.99),
                epsilon: Tensor<Scalar> = Tensor(0.001)) {
        self.axis = axis
        self.momentum = momentum
        self.scale = Tensor<Scalar>(ones: [featureCount])
        self.offset = Tensor<Scalar>(zeros: [featureCount])
        self.epsilon = epsilon
        self.runningMean = Parameter(Tensor(0))
        self.runningVariance = Parameter(Tensor(1))
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
    @noDerivative public let epsilon: Tensor<Scalar>

    /// Creates a layer normalization layer.
    public init(
        offset: Tensor<Scalar>,
        scale: Tensor<Scalar>,
        axis: Int,
        epsilon: Tensor<Scalar>
    ) {
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
    public init(featureCount: Int,
                axis: Int,
                epsilon: Tensor<Scalar> = Tensor(0.001)) {
        self.init(
            offset: Tensor(zeros: [featureCount]),
            scale: Tensor(ones: [featureCount]),
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
        let mean = input.mean(alongAxes: axis)
        let variance = input.variance(alongAxes: axis)
        let inv = rsqrt(variance + epsilon) * scale
        return (input - mean) * inv + offset
    }
}
