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

/// Returns the L1 loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func l1Loss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>,
    reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = { $0.sum() }
) -> Tensor<Scalar> {
    reduction(abs(expected - predicted))
}

/// Returns the L2 loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func l2Loss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>,
    reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = { $0.sum() }
) -> Tensor<Scalar> {
    reduction((expected - predicted).squared())
}

/// Returns the mean absolute error between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func meanAbsoluteError<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    l1Loss(predicted: predicted, expected: expected).mean()
}

/// Returns the mean squared error between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func meanSquaredError<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    l2Loss(predicted: predicted, expected: expected).mean()
}

/// Returns the mean squared logarithmic error between predictions and expectations.
///
/// - Note: Negative tensor entries will be clamped at `0` to avoid undefined
///   logarithmic behavior, as `log(_:)` is undefined for negative reals.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func meanSquaredLogarithmicError<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    let logPredicted = log(max(predicted, Tensor(0)) + 1)
    let logExpected = log(max(expected, Tensor(0)) + 1)
    return l2Loss(predicted: logPredicted, expected: logExpected).mean()
}

/// Returns the mean absolute percentage error between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func meanAbsolutePercentageError<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    100 * abs((expected - predicted) / abs(expected)).mean()
}

/// Returns the hinge loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func hingeLoss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>,
    reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = { $0.mean() }
) -> Tensor<Scalar> {
    reduction(max(Tensor(0), Tensor(1) - expected * predicted))
}

/// Returns the squared hinge loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func squaredHingeLoss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>,
    reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = { $0.mean() }
) -> Tensor<Scalar> {
    reduction(hingeLoss(predicted: predicted, expected: expected).squared())
}

/// Returns the hinge loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func categoricalHingeLoss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>,
    reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = { $0.mean() }
) -> Tensor<Scalar> {
    let positive = (expected * predicted).sum(alongAxes: -1)
    let negative = ((Tensor(1) - expected) * predicted).max(alongAxes: -1)
    return reduction(max(Tensor(0), negative - positive + Tensor(1)))
}

/// Returns the logarithm of the hyperbolic cosine of the error between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func logCoshLoss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>,
    reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = { $0.mean() }
) -> Tensor<Scalar> {
    let x = predicted - expected
    return reduction(x + softplus(Tensor(-2) * x) - log(Tensor(2)))
}

/// Returns the Poisson loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func poissonLoss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>,
    reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = { $0.mean() }
) -> Tensor<Scalar> {
    reduction(predicted - expected * log(predicted))
}

/// Returns the Kullback-Leibler divergence (KL divergence) between between expectations and predictions.
/// Given two distributions `p` and `q`, KL divergence computes `p * log(p / q)`.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func kullbackLeiblerDivergence<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>,
    expected: Tensor<Scalar>,
    reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = { $0.sum() }
) -> Tensor<Scalar> {
    reduction(expected * log(expected / predicted))
}

/// Returns the softmax cross entropy (categorical cross entropy) between logits and labels.
///
/// - Parameters:
///   - logits: One-hot encoded outputs from a neural network.
///   - labels: Indices (zero-indexed) of the correct outputs.
@differentiable(wrt: logits)
public func softmaxCrossEntropy<Scalar: TensorFlowFloatingPoint>(
    logits: Tensor<Scalar>,
    labels: Tensor<Int32>,
    reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = { $0.mean() }
) -> Tensor<Scalar> {
    reduction(softmaxCrossEntropyHelper(logits: logits, labels: labels))
}

@inlinable
@differentiable(wrt: logits, vjp: _vjpSoftmaxCrossEntropyHelper(logits:labels:))
func softmaxCrossEntropyHelper<Scalar: TensorFlowFloatingPoint>(
    logits: Tensor<Scalar>,
    labels: Tensor<Int32>
) -> Tensor<Scalar> {
    Raw.sparseSoftmaxCrossEntropyWithLogits(features: logits, labels: labels).loss
}

@inlinable
func _vjpSoftmaxCrossEntropyHelper<Scalar: TensorFlowFloatingPoint>(
    logits: Tensor<Scalar>,
    labels: Tensor<Int32>
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> Tensor<Scalar>) {
    let (loss, grad) = Raw.sparseSoftmaxCrossEntropyWithLogits(features: logits, labels: labels)
    return (loss, { $0.expandingShape(at: -1) * grad })
}

/// Returns the softmax cross entropy (categorical cross entropy) between logits and labels.
///
/// - Parameters:
///   - logits: Unscaled log probabilities from a neural network.
///   - probabilities: Probability values that correspond to the correct output. Each row must be a
///                    valid probability distribution.
@differentiable(wrt: logits)
public func softmaxCrossEntropy<Scalar: TensorFlowFloatingPoint>(
    logits: Tensor<Scalar>,
    probabilities: Tensor<Scalar>,
    reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = { $0.mean() }
) -> Tensor<Scalar> {
    reduction(softmaxCrossEntropyHelper(logits: logits, probabilities: probabilities))
}

@inlinable
@differentiable(wrt: logits, vjp: _vjpSoftmaxCrossEntropyHelper(logits:probabilities:))
func softmaxCrossEntropyHelper<Scalar: TensorFlowFloatingPoint>(
    logits: Tensor<Scalar>,
    probabilities: Tensor<Scalar>
) -> Tensor<Scalar> {
    Raw.softmaxCrossEntropyWithLogits(features: logits, labels: probabilities).loss
}

@inlinable
func _vjpSoftmaxCrossEntropyHelper<Scalar: TensorFlowFloatingPoint>(
    logits: Tensor<Scalar>,
    probabilities: Tensor<Scalar>
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> Tensor<Scalar>) {
    let (loss, grad) = Raw.softmaxCrossEntropyWithLogits(features: logits, labels: probabilities)
    return (loss, { $0.expandingShape(at: -1) * grad })
}

/// Returns the sigmoid cross entropy (binary cross entropy) between logits and labels.
///
/// The reduction is reduced over all elements. If reduced over batch size is intended, please
/// consider to scale the loss.
///
/// - Parameters:
///   - logits: The unscaled output of a neural network.
///   - labels: Integer values that correspond to the correct output.
@differentiable(wrt: logits)
public func sigmoidCrossEntropy<Scalar: TensorFlowFloatingPoint>(
    logits: Tensor<Scalar>,
    labels: Tensor<Scalar>,
    reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = { $0.mean() }
) -> Tensor<Scalar> {
    // This numerically stable implementation is based on the TensorFlow Python API.
    let maxLogitsWithZero = max(logits, Tensor(0))
    return reduction(maxLogitsWithZero - logits * labels + log(1 + exp(-abs(logits))))
}
