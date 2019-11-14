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

// TODO: Remove this file after 0.4.

public extension Tensor where Scalar == Int32 {
    /// Creates a tensor with the specified shape, randomly sampling scalar values from a discrete
    /// uniform distribution.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - generator: Random number generator to use.
    ///   - lowerBound: The lower bound of the distribution.
    ///   - upperBound: The upper bound of the distribution.
    @available(*, deprecated, message: "This API will be removed after Swift for TensorFlow 0.4.")
    init<G: RandomNumberGenerator>(
        randomUniform shape: TensorShape,
        generator: inout G,
        lowerBound: Scalar = Scalar.min,
        upperBound: Scalar = Scalar.max
    ) {
        let dist = UniformIntegerDistribution<Scalar>(
            lowerBound: lowerBound,
            upperBound: upperBound)
        var scalars: [Scalar] = []
        for _ in 0 ..< shape.contiguousSize {
            scalars.append(dist.next(using: &generator))
        }
        self.init(shape: shape, scalars: scalars)
    }

    /// Creates a tensor with the specified shape, randomly sampling scalar values from a discrete
    /// uniform distribution, using the default random number generator.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - lowerBound: The lower bound of the distribution.
    ///   - upperBound: The upper bound of the distribution.
    @available(*, deprecated, message: "This API will be removed after Swift for TensorFlow 0.4.")
    init(
        randomUniform shape: TensorShape,
        lowerBound: Scalar = Scalar.min,
        upperBound: Scalar = Scalar.max
    ) {
        self.init(
            randomUniform: shape,
            generator: &Context.local.randomNumberGenerator,
            lowerBound: lowerBound,
            upperBound: upperBound)
    }
}

public extension Tensor where Scalar: BinaryFloatingPoint,
                              Scalar.RawSignificand: FixedWidthInteger {
    /// Creates a tensor with the specified shape, randomly sampling scalar values from a uniform
    /// distribution between `lowerBound` and `upperBound`.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - generator: Random number generator to use.
    ///   - lowerBound: The lower bound of the distribution.
    ///   - upperBound: The upper bound of the distribution.
    @available(*, deprecated, message: "This API will be removed after Swift for TensorFlow 0.4.")
    init<G: RandomNumberGenerator>(
        randomUniform shape: TensorShape,
        generator: inout G,
        lowerBound: Scalar = 0,
        upperBound: Scalar = 1
    ) {
        let dist = UniformFloatingPointDistribution<Scalar>(
            lowerBound: lowerBound,
            upperBound: upperBound)
        var scalars: [Scalar] = []
        for _ in 0 ..< shape.contiguousSize {
            scalars.append(dist.next(using: &generator))
        }
        let sample = Tensor(shape: shape, scalars: scalars)
        self = (upperBound - lowerBound) * sample + lowerBound
    }

    /// Creates a tensor with the specified shape, randomly sampling scalar values from a normal
    /// distribution.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - generator: Random number generator to use.
    ///   - mean: The mean of the distribution.
    ///   - standardDeviation: The standard deviation of the distribution.
    @available(*, deprecated, message: "This API will be removed after Swift for TensorFlow 0.4.")
    init<G: RandomNumberGenerator>(
        randomNormal shape: TensorShape,
        generator: inout G,
        mean: Scalar = 0,
        standardDeviation: Scalar = 1
    ) {
        let dist = NormalDistribution<Scalar>(mean: mean, standardDeviation: standardDeviation)
        var scalars: [Scalar] = []
        for _ in 0 ..< shape.contiguousSize {
            scalars.append(dist.next(using: &generator))
        }
        let sample = Tensor(shape: shape, scalars: scalars)
        self = standardDeviation * sample + mean
    }
}

// TODO: Can become fileprivate after the 0.4 release.
internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    static func glorot(
        fromStandardUniform randomUniform: __shared Tensor<Scalar>,
        shape: __shared TensorShape
    ) -> Tensor<Scalar> {
        let spatialDimCount = shape.count - 2
        let receptiveField = shape[0..<spatialDimCount].contiguousSize
        let fanIn = shape[shape.count - 2] * receptiveField
        let fanOut = shape[shape.count - 1] * receptiveField
        let minusOneToOne = 2 * randomUniform - 1
        return Scalar.sqrt(Scalar(6) / Scalar(fanIn + fanOut)) * minusOneToOne
    }
}

// TODO: Can become fileprivate after the 0.4 release.
internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    static func glorot(
        fromStandardNormal standardNormal: __shared Tensor<Scalar>,
        shape: __shared TensorShape
    ) -> Tensor<Scalar> {
        let spatialDimCount = shape.count - 2
        let receptiveField = shape[0..<spatialDimCount].contiguousSize
        let fanIn = shape[shape.count - 2] * receptiveField
        let fanOut = shape[shape.count - 1] * receptiveField
        let minusOneToOne = 2 * standardNormal - 1
        return Scalar.sqrt(Scalar(2) / Scalar(fanIn + fanOut)) * minusOneToOne
    }
}

public extension Tensor where Scalar: TensorFlowFloatingPoint {
    /// Performs Glorot uniform initialization for the specified shape, creating a tensor by
    /// randomly sampling scalar values from a uniform distribution between `-limit` and `limit`,
    /// where limit is `sqrt(6 / (fanIn + fanOut))` and `fanIn`/`fanOut` represent the number of
    /// input and output features multiplied by the receptive field if present.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - generator: Random number generator to use.
    @available(*, deprecated, message: "This API will be removed after Swift for TensorFlow 0.4.")
    init<G: RandomNumberGenerator>(glorotUniform shape: TensorShape, generator: inout G) {
        let uniform = Tensor(randomUniform: shape, generator: &generator)
        self = Tensor.glorot(fromStandardUniform: uniform, shape: shape)
    }

    /// Performs Glorot normal initialization for the specified shape, creating a tensor by
    /// randomly sampling scalar values from a uniform distribution between `-limit` and `limit`,
    /// where limit is `sqrt(2 / (fanIn + fanOut))` and `fanIn`/`fanOut` represent the number of
    /// input and output features multiplied by the receptive field if present.
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor.
    ///   - generator: Random number generator to use.
    @available(*, deprecated, message: "This API will be removed after Swift for TensorFlow 0.4.")
    init<G: RandomNumberGenerator>(glorotNormal shape: TensorShape, generator: inout G) {
        let normal = Tensor(randomNormal: shape, generator: &generator)
        self = Tensor.glorot(fromStandardNormal: normal, shape: shape)
    }
}

//===------------------------------------------------------------------------------------------===//
// Losses
//===------------------------------------------------------------------------------------------===//

/// Returns the L1 loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func l1Loss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    l1Loss(predicted: predicted, expected: expected, reduction: { $0.sum() })
}

/// Returns the L2 loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func l2Loss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    l2Loss(predicted: predicted, expected: expected, reduction: { $0.sum() })
}

/// Returns the hinge loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func hingeLoss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    hingeLoss(predicted: predicted, expected: expected, reduction: _mean)
}

/// Returns the squared hinge loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func squaredHingeLoss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    squaredHingeLoss(predicted: predicted, expected: expected, reduction: _mean)
}

/// Returns the categorical hinge loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func categoricalHingeLoss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    categoricalHingeLoss(predicted: predicted, expected: expected, reduction: _mean)
}

/// Returns the logarithm of the hyperbolic cosine of the error between predictions and
/// expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func logCoshLoss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    logCoshLoss(predicted: predicted, expected: expected, reduction: _mean)
}

/// Returns the Poisson loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func poissonLoss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    poissonLoss(predicted: predicted, expected: expected, reduction: _mean)
}

/// Returns the Kullback-Leibler divergence (KL divergence) between between expectations and
/// predictions. Given two distributions `p` and `q`, KL divergence computes `p * log(p / q)`.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func kullbackLeiblerDivergence<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    kullbackLeiblerDivergence(predicted: predicted, expected: expected, reduction: { $0.sum() })
}

/// Returns the softmax cross entropy (categorical cross entropy) between logits and labels.
///
/// - Parameters:
///   - logits: One-hot encoded outputs from a neural network.
///   - labels: Indices (zero-indexed) of the correct outputs.
@differentiable(wrt: logits)
public func softmaxCrossEntropy<Scalar: TensorFlowFloatingPoint>(
    logits: Tensor<Scalar>,
    probabilities: Tensor<Scalar>
) -> Tensor<Scalar> {
    softmaxCrossEntropy(logits: logits, probabilities: probabilities, reduction: _mean)
}

/// Returns the sigmoid cross entropy (binary cross entropy) between logits and labels.
/// - Parameters:
///   - logits: The unscaled output of a neural network.
///   - labels: Integer values that correspond to the correct output.
@differentiable(wrt: logits)
public func sigmoidCrossEntropy<Scalar: TensorFlowFloatingPoint>(
    logits: Tensor<Scalar>,
    labels: Tensor<Scalar>
) -> Tensor<Scalar> {
    sigmoidCrossEntropy(logits: logits, labels:labels, reduction: _mean)
}
