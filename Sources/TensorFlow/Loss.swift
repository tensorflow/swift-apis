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

/// Returns the L1 loss between predictions and labels.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - labels: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func l1Loss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>, expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    return abs(expected - predicted).sum()
}

/// Returns the L2 loss between predictions and labels.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - labels: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func l2Loss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>, expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    return (expected - predicted).squared().sum()
}

/// Returns the mean squared error between predictions and labels.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - labels: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func meanSquaredError<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>, expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    return (expected - predicted).squared().mean()
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
    predicted: Tensor<Scalar>, expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    let logPredicted = log(max(predicted, Tensor(0)) + 1)
    let logExpected = log(max(expected, Tensor(0)) + 1)
    return (logPredicted - logExpected).squared().mean()
}

/// Returns the mean absolute error between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func meanAbsoluteError<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>, expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    return abs(expected - predicted).mean()
}

/// Returns the mean absolute percentage error between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func meanAbsolutePercentageError<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>, expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    let diff = abs((expected - predicted) / abs(expected))
    return 100 * diff.mean()
}

/// Returns the hinge loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func hingeLoss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>, expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    return max(Tensor(1) - expected * predicted, Tensor(0)).mean()
}

/// Returns the cosine similarity between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: (predicted, expected))
public func cosineSimilarity<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>, expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    return -(expected * predicted).sum() /
        (sqrt(expected.squared().sum()) * sqrt(predicted.squared().sum()))
}

/// Returns the squared hinge loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func squaredHingeLoss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>, expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    return (max(Tensor(1) - expected * predicted, Tensor(0))).squared().mean()
}

/// Returns the hinge loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func categoricalHingeLoss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>, expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    let positive = (expected * predicted).sum()
    let negative = ((Tensor(1) - expected) * predicted).max()
    return max(Tensor(0), negative - positive + Tensor(1))
}

/// Returns the Poisson loss between predictions and expectations.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func poissonLoss<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>, expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    return (predicted - expected * log(predicted)).mean()
}

/// Returns the Kullback-Leibler divergence (KL divergence) between between expectations and predictions.
/// Given two distributions `p` and `q`, KL divergence computes `(p * log(p / q)).sum()`.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
@differentiable(wrt: predicted)
public func kullbackLeiblerDivergence<Scalar: TensorFlowFloatingPoint>(
    predicted: Tensor<Scalar>, expected: Tensor<Scalar>
) -> Tensor<Scalar> {
    return (expected * log(expected / predicted)).sum()
}

/// Returns the softmax cross entropy (categorical cross entropy) between logits and labels.
///
/// - Parameters:
///   - logits: One-hot encoded outputs from a neural network.
///   - labels: Indices (zero-indexed) of the correct outputs.
@differentiable(wrt: logits, vjp: _vjpSoftmaxCrossEntropy)
public func softmaxCrossEntropy<Scalar: TensorFlowFloatingPoint>(
    logits: Tensor<Scalar>, labels: Tensor<Int32>
) -> Tensor<Scalar> {
    return Raw.sparseSoftmaxCrossEntropyWithLogits(features: logits, labels: labels).loss.mean()
}

@usableFromInline
func _vjpSoftmaxCrossEntropy<Scalar: TensorFlowFloatingPoint>(
    logits: Tensor<Scalar>, labels: Tensor<Int32>
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> Tensor<Scalar>) {
    let (loss, grad) = Raw.sparseSoftmaxCrossEntropyWithLogits(features: logits, labels: labels)
    let batchSize = Tensor<Scalar>(logits.shapeTensor[0])
    return (loss.mean(), { v in (v / batchSize) * grad })
}

/// Returns the softmax cross entropy (categorical cross entropy) between logits and labels.
///
/// - Parameters:
///   - logits: Unscaled log probabilities from a neural network.
///   - probabilities: Probability values that correspond to the correct output. Each row must be a
///                    valid probability distribution.
@differentiable(wrt: logits, vjp: _vjpSoftmaxCrossEntropy)
public func softmaxCrossEntropy<Scalar: TensorFlowFloatingPoint>(
    logits: Tensor<Scalar>, probabilities: Tensor<Scalar>
) -> Tensor<Scalar> {
    return Raw.softmaxCrossEntropyWithLogits(features: logits, labels: probabilities).loss.mean()
}

@usableFromInline
func _vjpSoftmaxCrossEntropy<Scalar: TensorFlowFloatingPoint>(
    logits: Tensor<Scalar>, probabilities: Tensor<Scalar>
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> Tensor<Scalar>) {
    let (loss, grad) = Raw.softmaxCrossEntropyWithLogits(features: logits, labels: probabilities)
    let batchSize = Tensor<Scalar>(logits.shapeTensor[0])
    return (loss.mean(), { v in v / batchSize * grad })
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
    logits: Tensor<Scalar>, labels: Tensor<Scalar>
) -> Tensor<Scalar> {
    // This numerical stable implementation is based on tf.nn.sigmoid_cross_entropy_with_logits.

    let maxLogitsWithZero = max(logits, Tensor(0))
    let loss = maxLogitsWithZero - logits * labels + log(1 + exp(-abs(logits)))
    return loss.mean()
}
