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

/// Computes the mean squared error between predictions and labels.
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

/// Computes the mean absolute error between predictions and expectations.
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

/// Computes the softmax cross entropy (categorical cross entropy) between logits and labels.
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

/// Computes the softmax cross entropy (categorical cross entropy) between logits and labels.
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

/// Computes the sigmoid cross entropy (binary cross entropy) between logits and labels.
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
